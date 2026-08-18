# tree-gen-leaves-to-instances-script
A script for friggog/tree-gen blender addon that turns leaves/blossom mesh into single leaf and pointcloud with geometry nodes modifier that uses pointcloud as instances to scatter leaf mesh. Result mesh after realizing instances perfectly matches original leaves/blossom mesh.

To use the script, open it in "text" menu in blender, select leaves/blossom mesh you need and run script by pressing that "play" button on the top of menu.

Can produce both just instances or realized into mesh instances. Just instances may save around 20% of VRAM for render, but render time may increase, i haven't tested it properly, also just instances will use more VRAM than just leaves mesh if multiplying the tree via array/geometry nodes (and it will lag alot in preview mode), so by default instances are realized into mesh that looks 100% like leaves mesh before processing.
