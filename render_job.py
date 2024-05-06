import os
import json
import wgpu

##
class RenderJob(object):

    ##
    def __init__(
        self,
        device, 
        present_context,
        render_job_file_path,
        canvas_width,
        canvas_height,
        curr_render_jobs):

        self.output_size = canvas_width, canvas_height, 1

        self.present_context = present_context

        # render job file
        file = open(render_job_file_path, 'rb')
        file_content = file.read()
        file.close()
        render_job_dict = json.loads(file_content)
        
        self.uniform_data = None

        self.name = render_job_dict['Name']
        self.type = render_job_dict['Type']
        self.pass_type = render_job_dict['PassType']

        # shader file
        self.shader_path = os.path.join('shaders', render_job_dict['Shader'])
        file = open(self.shader_path, 'rb')
        file_content = file.read()
        file.close()
        shader_source = file_content.decode('utf-8')
        self.shader = device.create_shader_module(code=shader_source)

        # attachments
        attachment_info = render_job_dict["Attachments"]
        self.create_attachments(
            attachment_info,
            device,
            canvas_width, 
            canvas_height,
            curr_render_jobs)

        # pipeline data
        self.shader_resources = render_job_dict['ShaderResources']
        self.create_pipeline_data(
            shader_resource_dict = self.shader_resources, 
            device = device)

        # pipeline binding and layout
        self.init_pipeline_layout(
            shader_resource_dict = self.shader_resources, 
            device = device)

        # render pipeline
        self.depth_texture_view = None
        self.init_render_pipeline(
            render_job_dict = render_job_dict,
            device = device)

    ##
    def create_attachments(
        self,
        attachment_info,
        device,
        width,
        height,
        curr_render_jobs):

        self.attachments = {}
        self.attachment_views = {}
        self.attachment_formats = {}

        self.attachment_info = attachment_info

        # swap chain uses presentable texture
        if self.pass_type == 'Swap Chain' or self.pass_type == "Swap Chain Full Triangle":
            texture_size = width, height, 1
            attachment_format = self.present_context.get_preferred_format(device.adapter)
            attachment_name = self.name + ' Output'
            self.attachments[attachment_name] = None
            self.attachment_formats[attachment_name] = attachment_format
            self.attachment_info.append(
                {
                    'Name': attachment_name,
                    'Type': 'TextureOutput',
                    'ParentJobName': 'This',
                    'Format': 'bgra8unorm-srgb',
                }
            )
        
        # create attachments
        for info in attachment_info:
            attachment_name = info['Name']
            attachment_type = info['Type']
            
            attachment_scale_width = 1.0
            if 'ScaleWidth' in info:
                attachment_scale_width = info['ScaleWidth']

            attachment_scale_height = 1.0
            if 'ScaleHeight' in info:
                attachment_scale_height = info['ScaleHeight']

            attachment_width = int(width * attachment_scale_width)
            attachment_height = int(height * attachment_scale_height)

            # create texture for output texture
            if attachment_type == 'TextureOutput':
                texture_size = attachment_width, attachment_height, 1
                attachment_format_str = info['Format']
                
                attachment_format = wgpu.TextureFormat.rgba8unorm
                if attachment_format_str == 'rgba32float':
                    attachment_format = wgpu.TextureFormat.rgba32float
                elif attachment_format_str == 'bgra8unorm-srgb':
                    attachment_format = wgpu.TextureFormat.bgra8unorm_srgb

                self.attachments[attachment_name] = device.create_texture(
                    size = texture_size,
                    usage = wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC,
                    dimension = wgpu.TextureDimension.d2,
                    format = attachment_format,
                    mip_level_count = 1,
                    sample_count = 1
                )

                self.attachment_views[attachment_name] = self.attachments[attachment_name].create_view()
                self.attachment_formats[attachment_name] = attachment_format
            
            elif attachment_type == 'TextureInput':
                # input texture

                parent_job_name = info['ParentJobName']
                parent_attachment_name = info['Name']

                # find the parent job for this input
                parent_job = None
                for render_job in curr_render_jobs:
                    if render_job.name == parent_job_name:
                        parent_job = render_job
                        break
                
                assert(parent_job != None)
                assert(parent_attachment_name in parent_job.attachments)

                # set view and format, attachment = None signals that it's an input attachment
                new_attachment_name = parent_job_name + "-" + info['Name']
                self.attachments[new_attachment_name] = None
                self.attachment_views[new_attachment_name] = parent_job.attachment_views[parent_attachment_name]
                self.attachment_formats[new_attachment_name] = parent_job.attachment_formats[parent_attachment_name]

            


    ##
    def create_pipeline_data(
        self, 
        shader_resource_dict, 
        device):

        self.uniform_buffers = []
        self.textures = []
        self.texture_views = []

        # shader resources
        for shader_resource_entry in shader_resource_dict:
            shader_resource_type = shader_resource_entry['type']
            shader_resource_usage = shader_resource_entry['usage']
            if shader_resource_type == 'buffer':
                # shader buffer
                
                shader_resource_size = shader_resource_entry['size']

                usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.STORAGE
                if shader_resource_usage == 'read_only_storage' or shader_resource_usage == 'storage':
                    usage = wgpu.BufferUsage.STORAGE

                self.uniform_buffers.append(device.create_buffer(
                    size = shader_resource_size, 
                    usage = usage | wgpu.BufferUsage.COPY_DST
                ))
            elif shader_resource_type == 'texture2d':
                # shader texture

                texture_width = shader_resource_entry['width']
                texture_height = shader_resource_entry['height']
                texture_size = texture_width, texture_height, 1
                texture_format = shader_resource_entry['format']

                format = wgpu.TextureFormat.r8unorm

                texture = device.create_texture(
                    size=texture_size,
                    usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.TEXTURE_BINDING,
                    dimension=wgpu.TextureDimension.d2,
                    format=format,
                    mip_level_count=1,
                    sample_count=1
                )

                self.textures.append(texture)
                self.texture_views.append(self.textures[len(self.textures) - 1].create_view())

    ##
    def init_pipeline_layout(
        self,
        shader_resource_dict, 
        device):
        
        bind_group_index = 0

        print('"{}"'.format(self.name))

        # We always have two bind groups, so we can play distributing our
        # resources over these two groups in different configurations.
        bind_groups_entries = [[]]
        bind_groups_layout_entries = [[]]

        # attachment bindings at group 0
        num_input_attachments = 0
        for attachment_index in range(len(self.attachments)):
            
            key = list(self.attachment_views.keys())[attachment_index]
            attachment_info = self.attachment_info[attachment_index]

            # render target doesn't need binding
            if attachment_info['Type'] == 'TextureOutput':
                continue

            # binding group
            bind_group_info = {
                "binding": num_input_attachments,
                "resource": self.attachment_views[key]
            }
            bind_groups_entries[bind_group_index].append(bind_group_info)

            # binding layout
            bind_group_layout_info = {
                "binding": num_input_attachments,
                "visibility": wgpu.ShaderStage.FRAGMENT,
                "texture": {
                    "sample_type": wgpu.TextureSampleType.unfilterable_float,
                    "view_dimension": wgpu.TextureViewDimension.d2,
                }
            }
            bind_groups_layout_entries[bind_group_index].append(bind_group_layout_info)
    
            print('\ttexture: "{}" binding group: {}, binding: {}'.format(
                    key,
                    bind_group_index,
                    num_input_attachments))

            num_input_attachments += 1

        if num_input_attachments > 0:
            bind_group_index += 1
            bind_groups_entries.append([])
            bind_groups_layout_entries.append([])

        # create group bindings and group layout
        texture_index = 0
        binding_index = 0
        uniform_buffer_index = 0
        for shader_resource_entry in shader_resource_dict:
            
            # shader stage (vertex/fragment/compute)
            shader_stage = wgpu.ShaderStage.VERTEX
            if shader_resource_entry['shader_stage'] == 'fragment':
                shader_stage = wgpu.ShaderStage.FRAGMENT

            # usage
            usage = wgpu.BufferBindingType.uniform
            if shader_resource_entry['usage'] == 'storage':
                usage = wgpu.BufferBindingType.storage
            elif shader_resource_entry['usage'] == 'read_only_storage':
                usage = wgpu.BufferBindingType.read_only_storage

            # build binding group and binding layout
            if shader_resource_entry['type'] == 'buffer':
                # buffer

                # binding group
                bind_group_info = {
                    "binding": binding_index,
                    "resource": 
                    {
                        "buffer": self.uniform_buffers[uniform_buffer_index],
                        "offset": 0,
                        "size": shader_resource_entry['size'],
                    },
                }
                
                bind_groups_entries[bind_group_index].append(bind_group_info)

                # binding layout
                bind_group_layout_info = {
                    "binding": binding_index,
                    "visibility": shader_stage,
                    "buffer": { "type": usage},
                }
                bind_groups_layout_entries[bind_group_index].append(bind_group_layout_info)

                print('\tbuffer: "{}" binding group: {}, binding: {}, uniform index: {}, size: {}, visibility: {}, usage "{}"'.format(
                    shader_resource_entry['name'],
                    bind_group_index,
                    binding_index,
                    uniform_buffer_index,
                    shader_resource_entry['size'], 
                    shader_stage, 
                    usage))

                uniform_buffer_index += 1

            elif shader_resource_entry['type'] == 'texture2d':
                # texture

                # binding group
                bind_group_info = {
                    "binding": binding_index,
                    "resource": self.texture_views[texture_index]
                }
                bind_groups_entries[bind_group_index].append(bind_group_info)

                # binding layout
                bind_group_layout_info = {
                    "binding": binding_index,
                    "visibility": shader_stage,
                    "texture": {
                        "sample_type": wgpu.TextureSampleType.unfilterable_float,
                        "view_dimension": wgpu.TextureViewDimension.d2,
                    }
                }
                bind_groups_layout_entries[bind_group_index].append(bind_group_layout_info)

                print('\ttexture: "{}" binding group: {}, binding: {}, size: {}, visibility: {}, usage "{}"'.format(
                    shader_resource_entry['name'],
                    bind_group_index,
                    binding_index,
                    shader_resource_entry['size'], 
                    shader_stage, 
                    usage))

            binding_index += 1

        # create sampler for textures
        num_attachment_binding_groups = len(bind_groups_entries[0])
        if len(self.texture_views) > 0 or num_input_attachments > 0:
            self.sampler = device.create_sampler()

            bind_groups_entries[0].append(
                {
                    "binding": num_attachment_binding_groups,
                    "resource": self.sampler
                }
            )
            bind_groups_layout_entries[0].append(
                {
                    "binding": num_attachment_binding_groups,
                    "visibility": wgpu.ShaderStage.FRAGMENT,
                    "sampler": {
                        "type": wgpu.SamplerBindingType.non_filtering
                    }
                }
            )

            print('\tsampler: group: 0, binding: {}'.format(
                    num_attachment_binding_groups))

        # Create the wgou binding objects
        bind_group_layouts = []
        self.bind_groups = []

        for entries, layout_entries in zip(bind_groups_entries, bind_groups_layout_entries):
            bind_group_layout = device.create_bind_group_layout(entries=layout_entries)
            bind_group_layouts.append(bind_group_layout)
            self.bind_groups.append(
                device.create_bind_group(layout=bind_group_layout, entries=entries)
            )

        self.pipeline_layout = device.create_pipeline_layout(bind_group_layouts=bind_group_layouts)


    ##
    def init_render_pipeline(
        self,
        render_job_dict, 
        device):

        cull_mode_str = render_job_dict['RasterState']['CullMode']
        front_face_str = render_job_dict['RasterState']['FrontFace']
        depth_enabled_str = render_job_dict['DepthStencilState']['DepthEnable']
        depth_func_str = render_job_dict['DepthStencilState']['DepthFunc']

        # cull mode
        cull_mode = wgpu.CullMode.none
        if cull_mode_str == 'Front':
            cull_mode = wgpu.CullMode.front
        elif cull_mode_str == 'Back':
            cull_mode = wgpu.CullMode.back
        
        # front face
        front_face = wgpu.FrontFace.ccw
        if front_face_str == 'Clockwise':
            front_face = wgpu.FrontFace.cw

        # depth toggle
        depth_enabled = False
        if depth_enabled_str == "True":
            depth_enabled = True

        # depth comparison function
        depth_func = wgpu.CompareFunction.never
        if depth_func_str == "Less":
            depth_func = wgpu.CompareFunction.less
        elif depth_func_str == "LessEqual":
            depth_func = wgpu.CompareFunction.less_equal
        elif depth_func_str == "Greater":
            depth_func = wgpu.CompareFunction.greater
        elif depth_func_str == "GreaterEqual":
            depth_func = wgpu.CompareFunction.greater_equal
        elif depth_func_str == "Equal":
            depth_func = wgpu.CompareFunction.equal
        elif depth_func_str == "NotEqual":
            depth_func = wgpu.CompareFunction.not_equal
        elif depth_func_str == "Always":
            depth_func = wgpu.CompareFunction.always

        self.depth_enabled = depth_enabled

        # attachment info
        render_target_info = []
        for attachment_name in self.attachments:
            
            if self.attachments[attachment_name] is not None:
                attachment_format = self.attachment_formats[attachment_name]

                if attachment_format == 'rgba32float':
                    render_target_info.append(
                        {
                            "format": attachment_format,
                        }
                    )
                else:
                    render_target_info.append(
                        {
                            "format": attachment_format,
                            "blend": {
                                "alpha": (
                                    wgpu.BlendFactor.one,
                                    wgpu.BlendFactor.zero,
                                    wgpu.BlendOperation.add,
                                ),
                                "color": (
                                    wgpu.BlendFactor.one,
                                    wgpu.BlendFactor.zero,
                                    wgpu.BlendOperation.add,
                                ),
                            },
                        }
                    )

                print('\tattachment: "{}", format: {}'.
                    format(
                        attachment_name, 
                        attachment_format))

        # create render pipeline
        if depth_enabled == False:
            self.render_pipeline = device.create_render_pipeline(
                layout=self.pipeline_layout,
                vertex={
                    "module": self.shader,
                    "entry_point": "vs_main",
                    "buffers": [
                        {
                            "array_stride": 4 * 10,                  # stride in bytes between the elements, ie. sizeof(float) * (4[xyzw] + 2[uv]) 
                            "step_mode": wgpu.VertexStepMode.vertex,
                            "attributes": [
                                {
                                    "format": wgpu.VertexFormat.float32x4,
                                    "offset": 0,
                                    "shader_location": 0,
                                },
                                {
                                    "format": wgpu.VertexFormat.float32x2,
                                    "offset": 4 * 4,
                                    "shader_location": 1,
                                },
                                {
                                    "format": wgpu.VertexFormat.float32x4,
                                    "offset": 4 * 4 + 4 * 2,
                                    "shader_location": 2,
                                },
                            ],
                        },
                    ],
                },
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_list,
                    "front_face": front_face,
                    "cull_mode": cull_mode,
                },
                multisample=False,
                fragment={
                    "module": self.shader,
                    "entry_point": "fs_main",
                    "targets": render_target_info,
                },
            )
        else:
            self.render_pipeline = device.create_render_pipeline(
                layout=self.pipeline_layout,
                vertex={
                    "module": self.shader,
                    "entry_point": "vs_main",
                    "buffers": [
                        {
                            "array_stride": 4 * 18,                  # stride in bytes between the elements, ie. sizeof(float) * (4[xyzw] + 2[uv]) 
                            "step_mode": wgpu.VertexStepMode.vertex,
                            "attributes": [
                                {
                                    "format": wgpu.VertexFormat.float32x4,
                                    "offset": 0,
                                    "shader_location": 0,
                                },
                                {
                                    "format": wgpu.VertexFormat.float32x2,
                                    "offset": 4 * 4,
                                    "shader_location": 1,
                                },
                                {
                                    "format": wgpu.VertexFormat.float32x4,
                                    "offset": 4 * 4 + 4 * 2,
                                    "shader_location": 2,
                                },
                                {
                                    "format": wgpu.VertexFormat.float32x4,
                                    "offset": 4 * 4 + 4 * 2 + 4 * 4,
                                    "shader_location": 3,
                                },
                                {
                                    "format": wgpu.VertexFormat.float32x4,
                                    "offset": 4 * 4 + 4 * 2 + 4 * 4 + 4 * 4,
                                    "shader_location": 4,
                                },
                            ],
                        },
                    ],
                },
                primitive={
                    "topology": wgpu.PrimitiveTopology.triangle_list,
                    "front_face": front_face,
                    "cull_mode": cull_mode,
                },
                depth_stencil=
                {
                    "format": wgpu.TextureFormat.depth24plus,
                    "depth_write_enabled": depth_enabled,
                    "depth_compare": depth_func,
                },
                multisample=False,
                fragment={
                    "module": self.shader,
                    "entry_point": "fs_main",
                    "targets": render_target_info,
                },
            )

        print('\tpipeline: depth enabled: {}, depth func: {}, front face: {}, cull mode: {}, vertex buffer array stride: {}'.format(
            depth_enabled,
            depth_func,
            front_face,
            cull_mode,
            4 * 10
        ))

        # create depth texture if depth test is enabled
        self.depth_texture = None
        if depth_enabled == True:
            depth_texture_size = self.output_size
            self.depth_texture = device.create_texture(
                size=depth_texture_size,
                usage=wgpu.TextureUsage.COPY_DST | wgpu.TextureUsage.RENDER_ATTACHMENT,
                dimension=wgpu.TextureDimension.d2,
                format=wgpu.TextureFormat.depth24plus,
                mip_level_count=1,
                sample_count=1,
            )

            self.depth_texture_view = self.depth_texture.create_view()
