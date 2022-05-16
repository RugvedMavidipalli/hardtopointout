import numpy as np
import json
import math

my_transform = np.matrix([
        [0, -0.49999999999999994, 0.8660254037844387, 2.0],
        [1, 0, 0, 0],
        [0, 0.8660254037844387, 0.49999999999999994, 1.2666666666666668],
        [0, 0, 0, 1]])

shift_coords = np.matrix([
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]])

#shift into normal 4d matrices, nerf's 4d matrices are shifted
tran_to_normal4d = np.linalg.inv(shift_coords)

def transform():
	with open("transforms.json", "r") as outfile:
		data = json.load(outfile)
		for frame in data['frames']:
			if frame['file_path'][-6:]=='16.jpg':
				colmap_m = np.array(frame['transform_matrix'])

				break


	#breakpoint()
	tran_matrix = np.matmul(tran_to_normal4d * my_transform, np.linalg.inv(tran_to_normal4d * colmap_m))

	for ind in range(0, len(data['frames'])):
		#tran to normal 4d first, and then do the transformation
		tm = tran_matrix * tran_to_normal4d * np.array(data["frames"][ind]["transform_matrix"])
		#then trans back to nerf matrcies system.
		data["frames"][ind]["transform_matrix"] = (shift_coords * tm).tolist()
		print(data['frames'][ind]["file_path"], data["frames"][ind]["transform_matrix"])
	with open("new_transforms.json", 'w') as f:
		json.dump(data, f, indent=2)

