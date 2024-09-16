Animation matching with WGPU python renderer

-cube-example.py is the main running file
-render with ReSTiR path tracing with multi-bounces using irradiance cache for static object
-traditional shadow map, ambient occlusion and indirect diffuse using visiblity mask for animating meshes
-render_job.json contains all the passes with many disabled experimental passes
-use embree to build BVH on the cpu side. This requires the embree4.dll, python311.dll,
and tbb12.dll
-need separate asset files
