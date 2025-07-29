# 1D UV Tools

Blender add-on.

Tools for working with UV Maps in Blender.

Add-on functionality
-

**UV Pack Tile**

Pack current existed UV Map to tile from 0 to 1. Used faces centroids for packing to 0...1 tile.

With enabled "Pack to Cursor" option - pack along the cursor.

**Store Key - Fit to Tile**

- select 3 points on the uv.
- press "Store Key" to save their coordinates to memory
  - v0 - anchor point, for rotating around it
  - v1 - horizontal point, rotation makes until v0 - v1 became horizontal
  - v2 - scale point
- select the uv iceland
- press "Fit to Tile" to move the selection to the 0.0 point, rotate selection around anchor point by horizontal point and, optionally, scale selection by scale point. 

Current version
-
1.2.5.

Blender version
-
2.79

Version history
-
1.2.5 
- Texel Scale operator renamed to "Retexel"

1.2.4 
- Add shifting all uv-points for selected faces in "Texel Scale" operator

1.2.1 - 1.2.3
- Fixing bugs

1.2.0
- Implemented "Multy Sure UV" functional from 1D_Scripts v. 0.10.22
- Added "Texel Scale" functional

1.1.0
- "Store Key - Fit to Tile" functional added

1.0.0
- First release
