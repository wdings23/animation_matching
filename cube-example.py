"""
This example renders a simple textured rotating cube.
"""

# test_example = true

import time

from wgpu.gui.auto import WgpuCanvas, run
import wgpu
import numpy as np

from mat4 import *
from quat import *

from render_job import *
from gltf_loader import *
from translate_keyframes import *

import logging

class MyCanvas(WgpuCanvas):
    def __init__(self, *, parent=None, size=None, title=None, max_fps=30, **kwargs):
        super().__init__(**kwargs)
        self.left_mouse_down = False
        self.diff_x = 0.0
        self.diff_y = 0.0
        self.last_x = 0.0
        self.last_y = 0.0

    def handle_event(self, event):
        if event['event_type'] == 'pointer_down':
            self.left_mouse_down = True
            self.last_x = event['x']
            self.last_y = event['y']

        elif event['event_type'] == 'pointer_up':
            self.left_mouse_down = False

        if event['event_type'] == 'pointer_move':
            if self.left_mouse_down == True:
                self.diff_x = event['x'] - self.last_x
                self.diff_y = event['y'] - self.last_y

                self.last_x = event['x']
                self.last_y = event['y']

class MyApp(object):
    
    ##
    def __init__(self):
        
        #logger = logging.getLogger(__name__)
        #logging.basicConfig(filename='d:\\test\\python-webgpu\\example.log', encoding='utf-8', level=logging.DEBUG)

        print("Available adapters on this system:")
        for a in wgpu.gpu.enumerate_adapters():
            print(a.summary)

        # Create a canvas to render to
        self.canvas = MyCanvas(size = (640, 480), title="wgpu cube")

        # Create a wgpu device
        self.adapter = wgpu.gpu.request_adapter(power_preference="high-performance")
        self.device = self.adapter.request_device()

        # Prepare present context
        self.present_context = self.canvas.get_context()
        self.render_texture_format = self.present_context.get_preferred_format(self.device.adapter)
        self.present_context.configure(device=self.device, format=self.render_texture_format)

        self.eye_z = -10.0
        self.angle_x = 0.0
        self.angle_y = 0.0

        # create render jobs
        self.load_render_jobs(path = 'd:\\test\\python-webgpu\\render-jobs\\render_jobs.json')

        self.animation_time = 0.0

    ##
    def load_render_jobs(
        self,
        path):

        file = open(path, 'rb')
        file_content = file.read()
        file.close()

        directory_end = path.rfind('\\')
        if directory_end == -1:
            directory_end = path.rfind('/')
        directory = path[:directory_end]

        self.render_job_info = json.loads(file_content.decode('utf-8'))

        self.render_jobs = []
        for info in self.render_job_info['Jobs']:
            
            file_name = info['Pipeline']

            full_render_job_path = os.path.join(directory, file_name)

            render_job = RenderJob(
                device = self.device,
                present_context = self.canvas.get_context(),
                render_job_file_path = full_render_job_path,
                canvas_width = int(self.canvas._logical_size[0]),
                canvas_height = int(self.canvas._logical_size[1]),
                curr_render_jobs = self.render_jobs)

            self.render_jobs.append(render_job)

    ##
    def init_data(self):

        # vertex and index buffers for full triangle pass triangle
        full_triangle_vertex_data = np.array(
            [
                [-1.0,  -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
                [-1.0, 3.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 1.0],
                [3.0,   -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            ],
            dtype = np.float32)

        full_triangle_index_data = np.array([
                [0, 1, 2]
            ],
            dtype = np.uint32).flatten()

        # Create vertex buffer, and upload data
        self.full_screen_triangle_vertex_buffer = self.device.create_buffer_with_data(
            data = full_triangle_vertex_data, 
            usage = wgpu.BufferUsage.VERTEX
        )

        # Create index buffer, and upload data
        self.full_screen_triangle_index_buffer = self.device.create_buffer_with_data(
            data = full_triangle_index_data, 
            usage=wgpu.BufferUsage.INDEX
        )

        mesh_positions, mesh_normals, mesh_texcoords, mesh_joint_indices, mesh_joint_weights, mesh_triangle_indices, rig, keyframe_channels, joint_to_node_mappings, inverse_bind_matrix_data = load_gltf('d:\\test\\python-webgpu\\assets\\walk.gltf')
        
        assert(len(mesh_positions[0]) == len(mesh_texcoords[0]))
        assert(len(mesh_positions[0]) == len(mesh_normals[0]))
        assert(len(mesh_positions[0]) == len(mesh_joint_indices[0]))
        assert(len(mesh_positions[0]) == len(mesh_joint_weights[0]))

        mesh_index = 0

        self.rig = None
        self.keyframe_channels = None
        
        translated_keyframe_channels = translate_keyframe_channels2(
            src_file_path = 'd:\\test\\mediapipe\\animations2\\38_03.gltf',
            dest_file_path = 'd:\\test\\python-webgpu\\assets\\walk.gltf'
        )

        # test test test
        # joint_positions = {}
        # joint_rotations = {}
        # for root_joint in rig.root_joints:
        #     traverse_rig(
        #         curr_joint = root_joint,
        #         rig = rig,
        #         keyframe_channels = keyframe_channels,
        #         time = 0.875,
        #         joint_positions = joint_positions,
        #         joint_rotations = joint_rotations,
        #         root_joint_name = root_joint.name)
        # for key in translated_keyframe_channels:
        #     position = float3(
        #         rig.joint_dict[key].total_matrix.entries[3],
        #         rig.joint_dict[key].total_matrix.entries[7],
        #         rig.joint_dict[key].total_matrix.entries[11])
        #     color = float3(0.0, 255.0, 0.0)
        #     print('draw_sphere([{}, {}, {}], 0.05, {}, {}, {}, 255, "{}") '.format(
        #         position.x,
        #         position.y,
        #         position.z,
        #         color.x,
        #         color.y,
        #         color.z,
        #         key))

        self.mesh_data = {
            'positions': mesh_positions, 
            'normals': mesh_normals, 
            'texcoords': mesh_texcoords, 
            'joint_indices': mesh_joint_indices, 
            'joint_weights': mesh_joint_weights, 
            'triangle_indices': mesh_triangle_indices, 
            'rig': rig,
            'keyframe_channels': keyframe_channels,
            'joint_to_node_mappings': joint_to_node_mappings,
            'inverse_bind_matrix_data': inverse_bind_matrix_data,
            'translated_keyframe_channels': translated_keyframe_channels,
        }

        mesh_xform_positions, mesh_xform_normals = self.test_skinning_transformations(
            joint_to_node_mappings = joint_to_node_mappings, 
            inverse_bind_matrix_data = inverse_bind_matrix_data,
            device = self.device)

        # vertex data
        vertex_data_3 = np.empty([0, 18], dtype = np.float32)
        for i in range(0, len(mesh_positions[mesh_index])):
            mesh_position = mesh_positions[mesh_index][i]
            texcoord = mesh_texcoords[mesh_index][i]
            normal = mesh_normals[mesh_index][i]

            mesh_xform_position = mesh_xform_positions[mesh_index][i]
            mesh_xform_normal = mesh_xform_normals[mesh_index][i]

            if len(mesh_joint_indices) > 0.0:
                joint_indices = mesh_joint_indices[mesh_index][i]
                joint_weights = mesh_joint_weights[mesh_index][i]

                vertex_list = [
                    #mesh_xform_position.x, mesh_xform_position.y, mesh_xform_position.z, 1.0,
                    mesh_position[0], mesh_position[1], mesh_position[2], 1.0, 
                    texcoord[0], texcoord[1], 
                    #mesh_xform_normal.x, mesh_xform_normal.y, mesh_xform_normal.z, 1.0,
                    normal[0], normal[1], normal[2], 1.0,
                    joint_weights[0], joint_weights[1], joint_weights[2], joint_weights[3],
                    float(joint_indices[0]), float(joint_indices[1]), float(joint_indices[2]), float(joint_indices[3]),
                ]
                self.rig = rig
                self.keyframe_channels = keyframe_channels
            else:
                vertex_list = [
                    mesh_position[0], mesh_position[1], mesh_position[2], 1.0, 
                    texcoord[0], texcoord[1], 
                    normal[0], normal[1], normal[2], 1.0
                ]
                
            array = np.array([np.array(vertex_list)], dtype = np.float32)
            vertex_data_3 = np.concatenate((vertex_data_3, array))
        
        # triangle index data
        index_data_3 = np.array([], dtype = np.uint32)
        for i in range(len(mesh_triangle_indices[mesh_index])):
            index = np.uint32(mesh_triangle_indices[mesh_index][i][0])
            index_data_3 = np.append(index_data_3, index)
            assert(index < len(mesh_positions[mesh_index]))


        self.index_size = index_data_3.size

        # Create vertex buffer, and upload data
        self.vertex_buffer = self.device.create_buffer_with_data(
            data = vertex_data_3, 
            usage = wgpu.BufferUsage.VERTEX
        )

        # Create index buffer, and upload data
        self.index_buffer = self.device.create_buffer_with_data(
            data = index_data_3, 
            usage=wgpu.BufferUsage.INDEX
        )

        

    ##
    def init_draw(self):
        self.canvas.request_draw(self.draw_frame2)

    ##
    def update_camera(
        self,
        eye_position,
        look_at,
        up,
        view_width,
        view_height):

        view_matrix = float4x4.view_matrix(
            eye_position = eye_position, 
            look_at = look_at, 
            up = up)
        perspective_projection_matrix = float4x4.perspective_projection_matrix(
            field_of_view = math.pi * 0.25,
            view_width = view_width,
            view_height = view_height,
            far = 100.0,
            near = 1.0)

        return perspective_projection_matrix * view_matrix

    ##
    def update_skinning_matrices(
        self,
        joint_to_node_mappings, 
        inverse_bind_matrix_data,
        animation_time):

        rig = self.mesh_data['rig']
        #keyframe_channels = self.mesh_data['keyframe_channels']
        keyframe_channels = self.mesh_data['translated_keyframe_channels']

        joint_positions = {}
        joint_rotations = {}
        for root_joint in rig.root_joints:
            traverse_rig(
                curr_joint = root_joint,
                rig = rig,
                keyframe_channels = keyframe_channels,
                time = animation_time,
                joint_positions = joint_positions,
                joint_rotations = joint_rotations,
                root_joint_name = root_joint.name)

        joints = rig.joints

        # skin matrix => total matrix from animation key frames * inverse bind matrix
        skin_matrices = []
        for joint_index in range(len(joints)):
            node_index = joint_to_node_mappings[joint_index]

            joint = joints[node_index]

            inverse_bind_matrix_from_gltf = inverse_bind_matrix_data[joint_index]
            inverse_bind_matrix = float4x4(
                [
                    inverse_bind_matrix_from_gltf[0], inverse_bind_matrix_from_gltf[4], inverse_bind_matrix_from_gltf[8], inverse_bind_matrix_from_gltf[12],
                    inverse_bind_matrix_from_gltf[1], inverse_bind_matrix_from_gltf[5], inverse_bind_matrix_from_gltf[9], inverse_bind_matrix_from_gltf[13],
                    inverse_bind_matrix_from_gltf[2], inverse_bind_matrix_from_gltf[6], inverse_bind_matrix_from_gltf[10], inverse_bind_matrix_from_gltf[14],
                    inverse_bind_matrix_from_gltf[3], inverse_bind_matrix_from_gltf[7], inverse_bind_matrix_from_gltf[11], inverse_bind_matrix_from_gltf[15],
                ]
            )

            anim_matrix = joint.total_matrix * inverse_bind_matrix
            skin_matrices.append(anim_matrix)

        self.mesh_data['skin_matrices'] = skin_matrices

    ##
    def update_skinning_uniform_data(
        self,
        joint_to_node_mappings,
        inverse_bind_matrix_data,
        device,
        animation_time):

        self.update_skinning_matrices(
            joint_to_node_mappings = joint_to_node_mappings, 
            inverse_bind_matrix_data = inverse_bind_matrix_data,
            animation_time = animation_time
        )

        skin_matrices = self.mesh_data['skin_matrices']

        # buffer for the skin matrices
        matrix_index = 0
        num_matrix_batches = int(math.ceil(len(skin_matrices) / 16))
        self.uniform_skin_matrix_buffers = [] * num_matrix_batches
        self.uniform_skin_matrix_data = [] * num_matrix_batches
        for i in range(num_matrix_batches):
            if matrix_index >= len(skin_matrices):
                break

            skin_matrices_buffer_data = np.empty(
                [0, 16],
                dtype = np.float32
            )
            for j in range(16):
                if matrix_index >= len(skin_matrices):
                    break

                array = np.array([np.array(skin_matrices[matrix_index].entries)], dtype = np.float32)
                skin_matrices_buffer_data = np.concatenate((skin_matrices_buffer_data, array))

                matrix_index += 1

            self.uniform_skin_matrix_buffers.append(
                device.create_buffer_with_data(
                    data = skin_matrices_buffer_data, 
                    usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_SRC)
            )

            self.uniform_skin_matrix_data.append(skin_matrices_buffer_data)

    ##
    def test_skinning_transformations(
        self,
        joint_to_node_mappings, 
        inverse_bind_matrix_data,
        device):

        vertex_positions = self.mesh_data['positions']
        vertex_normals = self.mesh_data['normals']
        vertex_texcoords = self.mesh_data['texcoords']
        vertex_joint_indices = self.mesh_data['joint_indices'] 
        vertex_joint_weights = self.mesh_data['joint_weights']

        self.update_skinning_matrices(
            joint_to_node_mappings = joint_to_node_mappings, 
            inverse_bind_matrix_data = inverse_bind_matrix_data,
            animation_time = 0.2
        )

        skin_matrices = self.mesh_data['skin_matrices']

        # buffer for the skin matrices
        matrix_index = 0
        num_matrix_batches = int(math.ceil(len(skin_matrices) / 16))
        self.uniform_skin_matrix_buffers = [] * num_matrix_batches
        self.uniform_skin_matrix_data = [] * num_matrix_batches
        for i in range(num_matrix_batches):
            if matrix_index >= len(skin_matrices):
                break

            skin_matrices_buffer_data = np.empty(
                [0, 16],
                dtype = np.float32
            )
            for j in range(16):
                if matrix_index >= len(skin_matrices):
                    break

                array = np.array([np.array(skin_matrices[matrix_index].entries)], dtype = np.float32)
                skin_matrices_buffer_data = np.concatenate((skin_matrices_buffer_data, array))

                matrix_index += 1

            self.uniform_skin_matrix_buffers.append(
                device.create_buffer_with_data(
                    data = skin_matrices_buffer_data, 
                    usage = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_SRC)
            )

            self.uniform_skin_matrix_data.append(skin_matrices_buffer_data)
        
        num_meshes = len(vertex_positions)
        mesh_xform_positions = []
        mesh_xform_normals = []

        # transform vertices with skinning matrix
        for mesh_index in range(num_meshes):
            num_vertex_positions = len(vertex_positions[mesh_index])

            xform_positions = [float3(0.0, 0.0, 0.0)] * num_vertex_positions
            xform_normals = [float3(0.0, 0.0, 0.0)] * num_vertex_positions

            positions = vertex_positions[mesh_index]
            normals = vertex_normals[mesh_index]
            texcoords = vertex_texcoords[mesh_index]
            joint_indices = vertex_joint_indices[mesh_index]
            joint_weights = vertex_joint_weights[mesh_index]

            for vertex_index in range(num_vertex_positions):
                vertex_position = positions[vertex_index]
                vertex_normal = normals[vertex_index]

                # blend transformed positions with joint weights    
                total_xform = float3(0.0, 0.0, 0.0)
                total_xform_normal = float3(0.0, 0.0, 0.0)
                total_weights = 0.0
                for i in range(4):
                    vertex_joint_index = joint_indices[vertex_index][i]
                    vertex_joint_weight = joint_weights[vertex_index][i]

                    skinning_matrix = skin_matrices[vertex_joint_index]
                    normal_matrix = float4x4([
                        skinning_matrix.entries[0], skinning_matrix.entries[1], skinning_matrix.entries[2], 0.0,
                        skinning_matrix.entries[4], skinning_matrix.entries[5], skinning_matrix.entries[6], 0.0,
                        skinning_matrix.entries[8], skinning_matrix.entries[9], skinning_matrix.entries[10], 0.0,
                        skinning_matrix.entries[12], skinning_matrix.entries[13], skinning_matrix.entries[14], 1.0,
                    ])

                    xform = skinning_matrix.apply(float3(vertex_position[0], vertex_position[1], vertex_position[2]))
                    xform_normal = normal_matrix.apply(float3(vertex_normal[0], vertex_normal[1], vertex_normal[2]))

                    total_xform = total_xform + float3(
                        xform.x * vertex_joint_weight,
                        xform.y * vertex_joint_weight,
                        xform.z * vertex_joint_weight)

                    total_xform_normal = total_xform_normal + float3(
                        xform_normal.x * vertex_joint_weight,
                        xform_normal.y * vertex_joint_weight,
                        xform_normal.z * vertex_joint_weight)

                    total_weights += vertex_joint_weight

                xform_positions[vertex_index] = total_xform
                xform_normals[vertex_index] = total_xform_normal
                
            mesh_xform_positions.append(xform_positions)
            mesh_xform_normals.append(xform_normals)
        
        return mesh_xform_positions, mesh_xform_normals

    ##
    def draw_frame2(self):

        # Update uniform transform
        eye_position = float3(0.0, 0.0, self.eye_z)
        look_at = float3(0.0, 0.0, 0.0)
        
        delta_x = (2.0 * math.pi) / 640.0
        delta_y = (2.0 * math.pi) / 480.0

        # update angle with mouse position delta
        self.angle_x += self.canvas.diff_x * delta_x
        self.angle_y += self.canvas.diff_y * delta_y

        self.canvas.diff_x = 0.0
        self.canvas.diff_y = 0.0

        if self.angle_x > 2.0 * math.pi:
            self.angle_x = -2.0 * math.pi 
        elif self.angle_x < -2.0 * math.pi:
            self.angle_x = 2.0 * math.pi 

        if self.angle_y >= math.pi * 0.5:
            self.angle_y = math.pi * 0.5
        elif self.angle_y <= -math.pi * 0.5:
            self.angle_y = -math.pi * 0.5
        
        # rotate eye position
        quat_x = quaternion.from_angle_axis(float3(0.0, 1.0, 0.0), self.angle_x)
        quat_y = quaternion.from_angle_axis(float3(1.0, 0.0, 0.0), self.angle_y)
        total_quat = quat_x * quat_y
        total_matrix = total_quat.to_matrix()
        xform_eye_position = total_matrix.apply(eye_position)

        # update camera with new eye position
        up_direction = float3(0.0, 1.0, 0.0)
        view_projection_matrix = self.update_camera(
            eye_position = xform_eye_position, 
            look_at = look_at,
            up = up_direction,
            view_width = self.canvas._logical_size[0],
            view_height = self.canvas._logical_size[1])

        # view projection uniform data
        self.render_jobs[0].uniform_data = np.array(
            [
                [view_projection_matrix.entries[0], view_projection_matrix.entries[1], view_projection_matrix.entries[2], view_projection_matrix.entries[3]],
                [view_projection_matrix.entries[4], view_projection_matrix.entries[5], view_projection_matrix.entries[6], view_projection_matrix.entries[7]],
                [view_projection_matrix.entries[8], view_projection_matrix.entries[9], view_projection_matrix.entries[10], view_projection_matrix.entries[11]],
                [view_projection_matrix.entries[12], view_projection_matrix.entries[13], view_projection_matrix.entries[14], view_projection_matrix.entries[15]]
            ],
            dtype = np.float32)

        joint_to_node_mappings = self.mesh_data['joint_to_node_mappings']
        inverse_bind_matrix_data = self.mesh_data['inverse_bind_matrix_data']

        # update skinning matrices
        self.update_skinning_uniform_data(
            joint_to_node_mappings = joint_to_node_mappings,
            inverse_bind_matrix_data = inverse_bind_matrix_data,
            device = self.device, 
            animation_time = self.animation_time
        )
    
        # current presentable swapchain texture
        current_present_texture = self.present_context.get_current_texture()

        # command encoder for data upload and render pass
        command_encoder = self.device.create_command_encoder()

        # upload view projection matrix uniform
        self.device.queue.write_buffer(
            buffer = self.render_jobs[0].uniform_buffers[0],
            buffer_offset = 0,
            data = self.render_jobs[0].uniform_data
        )

        # upload skin matrices
        for i in range(len(self.uniform_skin_matrix_buffers)):
            self.device.queue.write_buffer(
                buffer = self.render_jobs[0].uniform_buffers[i+1],
                buffer_offset = 0,
                data = self.uniform_skin_matrix_data[i]
            )

        for render_job_index in range(len(self.render_jobs)):

            render_job = self.render_jobs[render_job_index]

            # view for the depth texture
            if render_job.depth_texture is not None:
                current_depth_texture_view = render_job.depth_texture.create_view()

            # output color attachments for render pass, create views for the attachments
            color_attachments = []
            if render_job.pass_type == 'Swap Chain' or render_job.pass_type == 'Swap Chain Full Triangle':
                # swap chain job, use presentable texture and view

                attachment_view = current_present_texture.create_view()
                color_attachments.append({
                    'view': attachment_view,
                    'resolve_target': None,
                    'clear_value': (0, 0, 0.3, 0),
                    'load_op': wgpu.LoadOp.clear,
                    'store_op': wgpu.StoreOp.store
                })

            else:
                # regular job, use its output attachments

                for attachment_name in render_job.attachments:
                    attachment = render_job.attachments[attachment_name]
                    if attachment != None:
                        # valid output attachment

                        attachment_view = render_job.attachment_views[attachment_name]
                        color_attachments.append({
                            'view': attachment_view,
                            'resolve_target': None,
                            'clear_value': (0, 1, 0, 0),
                            'load_op': wgpu.LoadOp.clear,
                            'store_op': wgpu.StoreOp.store
                        })

            # setup and show render pass
            if render_job.depth_texture is not None:
                render_pass = command_encoder.begin_render_pass(
                    color_attachments = color_attachments,
                    depth_stencil_attachment = 
                    {
                        "view": current_depth_texture_view,
                        "depth_clear_value": 1.0,
                        "depth_load_op": wgpu.LoadOp.clear,
                        "depth_store_op": wgpu.StoreOp.store,
                        "depth_read_only": False,
                        "stencil_clear_value": 0,
                        "stencil_load_op": wgpu.LoadOp.clear,
                        "stencil_store_op": wgpu.StoreOp.discard,
                        "stencil_read_only": False,
                    }
                )
            else:
                render_pass = command_encoder.begin_render_pass(
                    color_attachments = color_attachments
                )

            # vertex and index buffer based on the pass type
            vertex_buffer = self.vertex_buffer
            index_buffer = self.index_buffer
            index_size = self.index_size 
            if render_job.pass_type == 'Swap Chain Full Triangle' or render_job.pass_type == 'Full Triangle':
                index_buffer = self.full_screen_triangle_index_buffer
                vertex_buffer = self.full_screen_triangle_vertex_buffer
                index_size = 3

            # render pass pipeline, vertex and index buffers, bind groups, and draw
            render_pass.set_pipeline(render_job.render_pipeline)
            render_pass.set_index_buffer(index_buffer, wgpu.IndexFormat.uint32)
            render_pass.set_vertex_buffer(0, vertex_buffer)
            for bind_group_id, bind_group in enumerate(render_job.bind_groups):
                render_pass.set_bind_group(bind_group_id, bind_group, [], 0, 99)
            render_pass.draw_indexed(index_size, 1, 0, 0, 0)
            render_pass.end()

        self.device.queue.submit([command_encoder.finish()])

        self.canvas.request_draw()

        keys = list(self.keyframe_channels.keys())
        last_time = self.keyframe_channels[keys[0]][2].times[1]

        self.animation_time += 0.01
        if self.animation_time >= last_time:
            self.animation_time = 0.0


##
if __name__ == "__main__":
    app = MyApp()
    app.init_data()
    app.init_draw()

    run()