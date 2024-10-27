set_xmakever("2.5.9")

add_requires("glfw", "spdlog", "eigen", "tinyobjloader", "yaml-cpp", "thrust")
add_requires("cuda", {configs={utils={"cusparse", "cusolver", "cublas"}}})
-- add_requires("cmake::Vulkan 1.3", {system = true})
add_requires("vulkansdk")
add_rules("mode.release", "mode.debug")
set_languages("cxx17")

target("ImGui")
    set_kind("static")
    add_includedirs("ext/imgui")
    add_packages("glfw", "vulkansdk", {public=true})
    add_files("ext/imgui/*.cpp", "ext/imgui/backends/*.cpp")

target("MultiGridSim")
    set_kind("binary")
    add_includedirs("src/engine", "src/common", "src/physics", "src/renderer", "ext/imgui")
    add_files("src/main.cpp", "src/*/*.cpp", "src/*/*/*.cpp", "src/physics/*/*.cu", "src/common/*.cu")
    add_deps("ImGui")
    add_cuflags("-rdc=true")
    add_cugencodes("native")
    add_packages("cuda", "thrust", "glfw", "spdlog", "eigen", "vulkansdk", "tinyobjloader", "yaml-cpp", {public=true})
    -- add_syslinks("cublas", "cusolver", "cusparse")
    if is_os("windows") then
        add_links("Advapi32", "Gdi32", "User32")
    end
    set_targetdir("bin")