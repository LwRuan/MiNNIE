echo "Vulkan sdk's location:"
echo $VULKAN_SDK
echo "create outputdir"
mkdir -p bin/shaders
echo "start compiling"
for file in shaders/*.vert
do
  output_file=`echo $file | sed -E "s/shaders\/(\S*)\.(\S*)/bin\/shaders\/\1_\2\.spv/"`
  echo compile $file
  $VULKAN_SDK/Bin/glslc.exe $file -o $output_file
done
for file in shaders/*.frag
do
  output_file=`echo $file | sed -E "s/shaders\/(\S*)\.(\S*)/bin\/shaders\/\1_\2\.spv/"`
  echo compile $file
  $VULKAN_SDK/Bin/glslc.exe $file -o $output_file
done
echo "finish compiling"
