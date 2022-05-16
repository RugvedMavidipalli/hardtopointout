from main import main
from colmap2nerforig import start_colmap
from run import runnerf
from coordinate_transform import transform
import bosdyn.client
import argparse
import sys


# def main(argv):
# 	"""Command line interface."""
# 	parser = argparse.ArgumentParser()
# 	completed = False
# 	while not completed:
# 		# start spot and take initial pictures
# 		# start_spot(parser, argv)
# 		# run colmap
# 		# start_colmap()
# 		# run nerf
# 		# runnerf(gui=True)
# 		# select next location to go to
#
#
#
#
#
# # spot start moving and taking images
# # start_spot(parser, argv)
# # once the spot has collected some images, we run colmap
# # start_colmap()
#
# # once we get the coord and images, we start running nerf
# # completed = False
# # while not completed:
# # 	# Start spot take
# # 	# start_spot(parser, argv)
# # 	runnerf(gui=True)
# # exit(0)
#
#
# def start_spot(parser, argv):
# 	bosdyn.client.util.add_base_arguments(parser)
# 	options = parser.parse_args(argv)
# 	try:
# 		main(options)
# 		return True
# 	except Exception as exc:  # pylint: disable=broad-except
# 		logger = bosdyn.client.util.get_logger()
# 		logger.exception("Threw an exception")
# 		return False


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--hostname', default='138.16.161.12')
	parser.add_argument('--cheat_images', action='store_true')
	parser.add_argument('--cheat_images_path', default='data/cheat')
	parser.add_argument('--images_path', default='data/demo')
	parser.add_argument('--output_path', default='output')
	parser.add_argument("--aabb_scale", default=2, choices=["1","2","4","8","16"])
	parser.add_argument('--verbose', action='store_true')
	options = parser.parse_args()
	print(options.hostname)
	try:
		main(options)
	except Exception as exc:
		logger = bosdyn.client.util.get_logger()
		logger.exception("Threw an exception")

