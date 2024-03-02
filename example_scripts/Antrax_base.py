import numpy as np
import cv2
import os
import math
import time
import pandas as pd


def import_tracks(
	path,
	numFrames,
	export=False,
	min_track_length=0,
	strip_tail_frames=0,
	min_movement_px=0,
	VERBOSE=True,
	custom_order=None,
):
	"""
	Import all tracked paths (using blender motionExport.py) from specified folder and join them to a single array.
	Optionally, allows for export of created array containing all tracks into single .csv file

	:param path: location of exported .csv tracks
	:param numFrames: number of total analysed frames
	:param export: boolean, writes .csv file of all combined tracks if True

	:return: array of all imported tracks, row: frames, columns X / Y coordinates of individual track.
	         The first column consists of the frame numbers for easier readability if exported as a single file.
	"""
	if VERBOSE:
		print("importing tracks...")
	lst = os.listdir(path)
	number_files = len(lst)
	if VERBOSE:
		print("INFO: found a total of", number_files, "tracks...")

	files = []
	tracks = np.zeros(
		[numFrames + 1, 1 + number_files * 2]
	)  # create array for all tracks
	tracks[:, 0] = np.arange(
		start=1, stop=numFrames + 2, step=1, dtype=int
	)  # insert frame numbers

	imported = 0

	file_order = []

	if custom_order is not None:
		for file in custom_order:
			file_order.append(file)
			df = pd.read_csv(os.path.join(path, file), delimiter=";")

			track_temp = df[["frame", "x", "y"]].to_numpy()

			if strip_tail_frames > 0:
				track_temp = track_temp[:-strip_tail_frames]

			line_count = len(track_temp)

			if line_count >= min_track_length:
				# next, check if the track has moved sufficiently to be considered
				movement_temp_arr = np.array([i[1:3] for i in track_temp if i[1] != ""])
				movement_range = np.ptp(movement_temp_arr.astype(int), axis=0)

				if np.sqrt(np.sum(np.square(movement_range))) >= min_movement_px:
					tracks[
						track_temp[:, 0] - 1, imported * 2 + 1 : imported * 2 + 3
					] = track_temp[:, 1:]

					if VERBOSE:
						print(
							"INFO: imported", str(file), f" with {line_count} points."
						)
					imported += 1
				else:
					if VERBOSE:
						print(
							"INFO: excluded",
							str(file),
							"for insufficient movement points.",
						)
			else:
				if VERBOSE:
					print("INFO: excluded", str(file), f" with {line_count} points.")
	else:
		# r=root, d=directories, f = files
		for r, d, f in os.walk(path):
			for file in f:
				file_order.append(file)
				df = pd.read_csv(os.path.join(path, file), delimiter=";")

				track_temp = df[["frame", "x", "y"]].to_numpy()

				if strip_tail_frames > 0:
					track_temp = track_temp[:-strip_tail_frames]

				line_count = len(track_temp)

				if line_count >= min_track_length:
					# next, check if the track has moved sufficiently to be considered
					movement_temp_arr = np.array(
						[i[1:3] for i in track_temp if i[1] != ""]
					)
					movement_range = np.ptp(movement_temp_arr.astype(int), axis=0)

					if np.sqrt(np.sum(np.square(movement_range))) >= min_movement_px:
						tracks[
							track_temp[:, 0] - 1, imported * 2 + 1 : imported * 2 + 3
						] = track_temp[:, 1:]

						if VERBOSE:
							print(
								"INFO: imported",
								str(file),
								f" with {line_count} points.",
							)
						imported += 1
					else:
						if VERBOSE:
							print(
								"INFO: excluded",
								str(file),
								"for insufficient movement points.",
							)
				else:
					if VERBOSE:
						print(
							"INFO: excluded", str(file), f" with {line_count} points."
						)

	tracks = tracks.astype(int)
	if export:
		export_path = path + "_all_tracks.csv"
		np.savetxt(export_path, tracks, delimiter=",")

	if VERBOSE:
		print(
			"\nSuccessfully combined the tracks of",
			imported,
			"individuals for training and display!",
		)
	# trim array to only contain valid tracks
	return tracks[:, : 1 + imported * 2], file_order


def display_video(
	cap, tracks, show=(0, math.inf), scale=1.0, focus_track=None, export_video=False
):
	"""
	Function displays imported footage with tracking results as overlay

	:param cap: Imported video file
	:param tracks: all imported tracks as a single array, created with import_tracks
	:param show: tuple of desired displayed frames
	:param scale: single float to up- or downscale resolution of display
	"""
	tracks = (scale * tracks).astype(int)  # rescale pixel values of tracks
	# frame counter
	frame_num = show[0]

	# define the size of each tracking rectangle
	target_size = 100 * scale

	# get frame rate of imported footage
	fps = cap.get(cv2.CAP_PROP_FPS)

	# fix the seed for the same set of randomly assigned colours for each track
	np.random.seed(seed=0)
	colours = np.random.randint(
		low=0, high=255, size=((math.floor(((tracks.shape[1]) - 1) / 2)), 3)
	)

	print("\nDisplaying tracked footage!\npress 'q' to end display")

	# skip to desired start frame
	# Property identifier of cv2.CV_CAP_PROP_POS_FRAMES is 1, thus the first entry is 1
	cap.set(1, show[0])

	# set font from info display on frame
	font = cv2.FONT_HERSHEY_SIMPLEX

	if export_video:
		width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
		height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		out = cv2.VideoWriter(
			"ID" + str(focus_track) + "_start_frame_" + str(show[0]) + "_output.avi",
			fourcc,
			int(fps),
			(int(width), int(height)),
		)

	while True:  # run until no more frames are available
		time_prev = time.time()
		# return single frame (ret = boolean, frame = image)
		ret, frame = cap.read()
		if not ret:
			break

		# scale down the video
		new_height = int(np.shape(frame)[0] * scale)
		new_width = int(np.shape(frame)[1] * scale)
		frame = cv2.resize(frame, (new_width, new_height))

		if focus_track is not None:
			track = focus_track
			target_centre = np.asarray(
				[tracks[frame_num, track * 2 + 1], tracks[frame_num, track * 2 + 2]]
			)

			# invert y-axis, to fit openCV convention ( lower left -> (x=0,y=0) )
			target_centre[1] = new_height - target_centre[1]
			# define the starting and ending point of the bounding box rectangle, defined by "target_size"
			px_start = target_centre - np.asarray(
				[math.floor(target_size / 2), math.floor(target_size / 2)]
			)
			px_end = target_centre + np.asarray(
				[math.floor(target_size / 2), math.floor(target_size / 2)]
			)
			# draw the defined rectangle of the track on top of the frame
			cv2.rectangle(
				frame,
				(px_start[0], px_start[1]),
				(px_end[0], px_end[1]),
				(
					int(colours[track, 0]),
					int(colours[track, 1]),
					int(colours[track, 2]),
				),
				2,
			)
			# write out track number of each active track
			cv2.putText(
				frame,
				"track: " + str(track),
				(
					int(target_centre[0] - target_size / 2),
					int(target_centre[1] - target_size / 2 - 10),
				),
				font,
				0.3,
				(
					int(colours[track, 0]),
					int(colours[track, 1]),
					int(colours[track, 2]),
				),
				1,
				cv2.LINE_AA,
			)
		else:
			# iterate through all columns and draw rectangles for all non 0 values
			for track in range(math.floor(((tracks.shape[1]) - 1) / 2)):
				if tracks[frame_num, track * 2 + 1] != 0:
					# the tracks are read as centres
					target_centre = np.asarray(
						[
							tracks[frame_num, track * 2 + 1],
							tracks[frame_num, track * 2 + 2],
						]
					)

					# invert y axis, to fit openCV convention ( lower left -> (x=0,y=0) )
					target_centre[1] = new_height - target_centre[1]
					# define the starting and ending point of the bounding box rectangle, defined by "target_size"
					px_start = target_centre - np.asarray(
						[math.floor(target_size / 2), math.floor(target_size / 2)]
					)
					px_end = target_centre + np.asarray(
						[math.floor(target_size / 2), math.floor(target_size / 2)]
					)
					# draw the defined rectangle of the track on top of the frame
					cv2.rectangle(
						frame,
						(px_start[0], px_start[1]),
						(px_end[0], px_end[1]),
						(
							int(colours[track, 0]),
							int(colours[track, 1]),
							int(colours[track, 2]),
						),
						2,
					)
					# write out track number of each active track
					cv2.putText(
						frame,
						"track: " + str(track),
						(
							int(target_centre[0] - target_size / 2),
							int(target_centre[1] - target_size / 2 - 10),
						),
						font,
						0.3,
						(
							int(colours[track, 0]),
							int(colours[track, 1]),
							int(colours[track, 2]),
						),
						1,
						cv2.LINE_AA,
					)

		cv2.putText(
			frame,
			"frame: " + str(frame_num),
			(int(new_width / 2) - 100, 35),
			font,
			0.8,
			(255, 255, 255),
			1,
			cv2.LINE_AA,
		)
		cv2.imshow("original frame", frame)

		if export_video:
			out.write(frame)

		if frame_num > show[1]:
			break

		# enforce constant frame rate during display
		time_to_process = (
			time.time() - time_prev
		)  # compute elapsed time to enforce constant frame rate (if possible)
		if time_to_process < 1 / fps:
			time.sleep((1 / fps) - time_to_process)

		# press q to quit, i.e. exit the display
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

		frame_num += 1

	cv2.destroyAllWindows()

	if export_video:
		out.release()

	# always reset frame from capture at the end to avoid incorrect skips during access
	cap.set(1, 0)

	print("\nReached last frame of specified video or ended by user input.\n")


class BarGraph:
	def __init__(self, width=600, height=720, num_classes=5, lightmode=False):
		self.height = height
		self.width = width
		self.lightmode = lightmode
		if self.lightmode:
			self.graph = np.ones((height, width, 3), np.uint8) * 255
		else:
			self.graph = np.zeros((height, width, 3), np.uint8)
		self.num_classes = num_classes
		self.class_count = np.zeros(num_classes)

	def update(self, classes):
		unique, counts = np.unique(classes, return_counts=True)
		self.class_count = np.zeros(self.num_classes)
		for u, c in zip(unique, counts):
			self.class_count[u] = c

		if self.lightmode:
			new_graph = np.ones((self.height, self.width, 3), np.uint8) * 255
		else:
			new_graph = np.zeros((self.height, self.width, 3), np.uint8)

		colours = [
			[152, 181, 45],
			[181, 131, 45],
			[181, 52, 45],
			[181, 45, 118],
			[165, 45, 181],
		]

		if self.lightmode:
			fill = [
				[194, 232, 58],
				[232, 168, 58],
				[232, 67, 58],
				[232, 58, 151],
				[212, 58, 232],
			]
		else:
			fill = [
				[109, 130, 33],
				[130, 94, 33],
				[130, 37, 33],
				[130, 33, 85],
				[119, 33, 130],
			]

		num_occurences = len(classes)
		if num_occurences > 0:
			for c, cl in enumerate(self.class_count):
				# Using cv2.rectangle() method
				# Draw a rectangle with blue line borders of thickness of 2 px
				elem_height = cl / num_occurences
				start_point = int(c * self.width / self.num_classes), int(self.height)
				end_point = (
					int((c + 1) * self.width / self.num_classes),
					int(self.height * (1 - elem_height)),
				)
				new_graph = cv2.rectangle(
					new_graph, start_point, end_point, fill[c], cv2.FILLED
				)
				new_graph = cv2.rectangle(
					new_graph, start_point, end_point, colours[c], 4
				)

		self.graph = new_graph

	def get_graph(self):
		return self.graph


def compose_video_with_overlay(
	cap,
	tracks,
	size_classes=None,
	poses=None,
	show=(0, math.inf),
	scale=1.0,
	constant_frame_rate=True,
	thresh=0.5,
	pose_point_size=3,
	video_name="Analysed video",
	DEBUG=False,
	WRITE_OUT=True,
	lightmode=False,
):
	"""
	Function displays imported footage with tracking results, poses, and class, as overlay
	For now, this happens at a fixed resolution and input videos will be rescaled to 720p

	:param cap: Imported video file
	:param tracks: all imported tracks as a single array, created with import_tracks
	:param poses: list of pose data frames (shape of [len(tracks), pose_df])
	:param show: tuple of desired displayed frames
	:param scale: single float to up- or downscale resolution of display
	"""
	if WRITE_OUT:
		if not os.path.exists("output_comp_" + video_name):
			os.mkdir("output_comp_" + video_name)

	tracks = (scale * tracks).astype(int)  # rescale pixel values of tracks
	# frame counter
	frame_num = show[0]

	# define the size of each tracking rectangle
	target_size = 128 * scale

	pose_display_size = 300 * scale

	# get frame rate of imported footage
	fps = cap.get(cv2.CAP_PROP_FPS)

	# colours = np.random.randint(low=0, high=255, size=((math.floor(((tracks.shape[1]) - 1) / 2)), 3))
	# colours correspond to the class of the animal (lager = "more red")
	colours = [
		[152, 181, 45],
		[181, 131, 45],
		[181, 52, 45],
		[181, 45, 118],
		[165, 45, 181],
	]

	# assign IDs to cutout displays
	box_id = [None, None, None, None, None, None, None, None]
	tracks_used = []
	# define locations for cutout displays
	box_coords = [
		[200, 100, 500, 400],
		[620, 100, 920, 400],
		[1040, 100, 1340, 400],
		[1040, 520, 1340, 820],
		[1040, 950, 1340, 1250],
		[1040, 1370, 1340, 1670],
		[1040, 1800, 1340, 2100],
		[1040, 2230, 1340, 2530],
	]

	# define location for centred video and plot
	video_coords = [200, 920, 520, 1080]
	plot_coords = [200, 920, 1920, 2530]

	full_frame_dims = [1420, 2600, 3]

	graph = BarGraph(width=600, height=720, num_classes=5, lightmode=lightmode)

	print("\nDisplaying tracked footage!\npress 'q' to end display")

	# skip to desired start frame
	# Property identifier of cv2.CV_CAP_PROP_POS_FRAMES is 1, thus the first entry is 1
	cap.set(1, show[0])

	# set font from info display on frame
	font = cv2.FONT_HERSHEY_DUPLEX

	if lightmode:
		line_colour = (0, 0, 0)
	else:
		line_colour = (255, 255, 255)

	while True:  # run until no more frames are available
		time_prev = time.time()
		# return single frame (ret = boolean, frame = image)
		ret, frame = cap.read()
		if not ret:
			break

		# scale down the video
		new_height = int(np.shape(frame)[0] * scale)
		new_width = int(np.shape(frame)[1] * scale)
		frame = cv2.resize(frame, (new_width, new_height))

		frame_clean = frame.copy()

		all_classes = []

		if lightmode:
			full_frame = np.ones(full_frame_dims, dtype=np.uint8) * 255
			full_frame_clean = np.ones(full_frame_dims, dtype=np.uint8) * 255
		else:
			full_frame = np.zeros(full_frame_dims, dtype=np.uint8)
			full_frame_clean = np.zeros(full_frame_dims, dtype=np.uint8)

		full_frame_clean[200:920, 520:1800, :] = frame_clean

		for bbox in box_coords:
			start_point = bbox[1], bbox[0]
			end_point = bbox[3], bbox[2]
			full_frame = cv2.rectangle(
				full_frame, start_point, end_point, line_colour, 5
			)

		vis_track = 0

		for track in range(math.floor(((tracks.shape[1]) - 1) / 2)):
			if tracks[frame_num, track * 2 + 1] != 0:
				# the tracks are read as centres
				target_centre = np.asarray(
					[tracks[frame_num, track * 2 + 1], tracks[frame_num, track * 2 + 2]]
				)

				# invert y-axis, to fit openCV convention ( lower left -> (x=0,y=0) )
				target_centre[1] = new_height - target_centre[1]
				# define the starting and ending point of the bounding box rectangle, defined by "target_size"
				px_start = target_centre - np.asarray(
					[math.floor(target_size / 2), math.floor(target_size / 2)]
				)
				px_end = target_centre + np.asarray(
					[math.floor(target_size / 2), math.floor(target_size / 2)]
				)
				# draw the defined rectangle of the track on top of the frame

				# TODO use constant location
				"""
                for t, temp_cutout in enumerate(box_id):
                    if temp_cutout is None and track not in tracks_used:
                        box_id[t] = track
                        tracks_used.append(track)
                """

				if vis_track < len(box_id):
					cutout = full_frame_clean[
						200 + px_start[1] : 200 + px_end[1],
						520 + px_start[0] : 520 + px_end[0],
						:,
					]
					try:
						cutout_resized = cv2.resize(cutout, (300, 300))
						full_frame[
							box_coords[vis_track][0] : box_coords[vis_track][2],
							box_coords[vis_track][1] : box_coords[vis_track][3],
							:,
						] = cutout_resized
					except cv2.error:
						pass

				if size_classes is not None:
					all_classes.append(size_classes[track])
					color_temp = (
						int(colours[size_classes[track]][0]),
						int(colours[size_classes[track]][1]),
						int(colours[size_classes[track]][2]),
					)

					# write out CLASS
					cv2.putText(
						frame,
						"CLASS: " + str(size_classes[track]),
						(
							int(target_centre[0] - target_size / 2 + 80),
							int(target_centre[1] - target_size / 2 - 10),
						),
						font,
						0.33,
						color_temp,
						1,
						cv2.LINE_AA,
					)

					if vis_track < len(box_id):
						cv2.putText(
							full_frame,
							"CLASS: " + str(size_classes[track]),
							(box_coords[vis_track][1], box_coords[vis_track][2] + 40),
							font,
							0.83,
							color_temp,
							2,
							cv2.LINE_AA,
						)

				else:
					color_temp = (
						int(colours[0][0]),
						int(colours[0][1]),
						int(colours[0][2]),
					)

				cv2.rectangle(
					frame,
					(px_start[0], px_start[1]),
					(px_end[0], px_end[1]),
					color_temp,
					2,
				)
				# write out track number of each active track
				cv2.putText(
					frame,
					"track: " + str(track),
					(
						int(target_centre[0] - target_size / 2),
						int(target_centre[1] - target_size / 2 - 10),
					),
					font,
					0.33,
					color_temp,
					1,
					cv2.LINE_AA,
				)

				if poses is not None:
					temp_pose = poses[track][np.where(poses[track][:, 0] == frame_num)]
					num_points = int((len(temp_pose[0]) - 1) / 3)
					for p in range(num_points):
						if temp_pose[0][p * 3 + 3] >= thresh:
							point = temp_pose[0][p * 3 + 1 : p * 3 + 3]
							# convert the location to the full frame equivalent
							# first down scale (from 300 to 128)
							point_rescaled = (point / 300) * 128
							# then add the track centre
							point_global = point_rescaled + px_start
							frame = cv2.circle(
								frame,
								(int(point_global[0]), int(point_global[1])),
								pose_point_size,
								(
									int(255 * p / num_points),
									int(255 - 255 * p / num_points),
									200,
								),
								-1,
							)

							if vis_track < len(box_id):
								point_cutout = point + [
									box_coords[vis_track][1],
									box_coords[vis_track][0],
								]
								full_frame = cv2.circle(
									full_frame,
									(int(point_cutout[0]), int(point_cutout[1])),
									pose_point_size * 2,
									(
										int(255 * p / num_points),
										int(255 - 255 * p / num_points),
										200,
									),
									-1,
								)

				vis_track += 1

		full_frame[200:920, 520:1800, :] = frame

		cv2.putText(
			full_frame,
			"file: " + video_name,
			(100, 80),
			font,
			1.6,
			line_colour,
			1,
			cv2.LINE_AA,
		)

		cv2.putText(
			full_frame,
			"live size-frequency",
			(1990, 160),
			font,
			1.4,
			line_colour,
			1,
			cv2.LINE_AA,
		)

		cv2.putText(
			full_frame,
			"frame: " + str(frame_num),
			(100, 160),
			font,
			1.6,
			line_colour,
			1,
			cv2.LINE_AA,
		)

		# updated and create bar graph of normalised size frequency distribution
		graph.update(all_classes)
		roi = full_frame[200:920, 1920:2520, :]  # final position
		roi[:] = graph.get_graph()
		# draw lines and add ticks
		full_frame = cv2.line(
			full_frame, (1900, 920), (2540, 920), color=line_colour, thickness=5
		)
		full_frame = cv2.line(
			full_frame, (1910, 930), (1910, 190), color=line_colour, thickness=5
		)
		cv2.putText(
			full_frame,
			"body length in mm",
			(2090, 1000),
			font,
			0.9,
			line_colour,
			1,
			cv2.LINE_AA,
		)
		cv2.putText(
			full_frame, "0.0", (1850, 920), font, 0.7, line_colour, 1, cv2.LINE_AA
		)
		full_frame = cv2.line(
			full_frame, (1900, 200), (1920, 200), color=line_colour, thickness=3
		)
		cv2.putText(
			full_frame, "0.5", (1850, 560), font, 0.7, line_colour, 1, cv2.LINE_AA
		)
		full_frame = cv2.line(
			full_frame, (1900, 560), (1920, 560), color=line_colour, thickness=3
		)
		cv2.putText(
			full_frame, "1.0", (1850, 200), font, 0.7, line_colour, 1, cv2.LINE_AA
		)
		five_class = [" 3.0", " 4.0", " 5.0", " 6.0", " 7.0"]
		for space, cl in enumerate(five_class):
			full_frame = cv2.line(
				full_frame,
				(1975 + space * 125, 905),
				(1975 + space * 125, 935),
				color=line_colour,
				thickness=3,
			)
			cv2.putText(
				full_frame,
				cl,
				(1945 + space * 125, 963),
				font,
				0.7,
				line_colour,
				1,
				cv2.LINE_AA,
			)

		if WRITE_OUT:
			cv2.imwrite(
				os.path.join(
					"output_comp_" + video_name,
					video_name + "_" + str(frame_num) + ".jpg",
				),
				full_frame,
			)

		if DEBUG:
			full_frame = cv2.resize(full_frame, (1300, 710))

		cv2.imshow("original frame", full_frame)

		if frame_num > show[1]:
			break

		if constant_frame_rate:
			# enforce constant frame rate during display
			time_to_process = (
				time.time() - time_prev
			)  # compute elapsed time to enforce constant frame rate
			if time_to_process < 1 / fps:
				time.sleep((1 / fps) - time_to_process)

		# press q to quit, i.e. exit the display
		if cv2.waitKey(1) & 0xFF == ord("q"):
			break

		frame_num += 1

	cv2.destroyAllWindows()

	# always reset frame from capture at the end to avoid incorrect skips during access
	cap.set(1, 0)

	print("\nReached last frame of specified video or ended by user input.\n")


def get_exact_frame(frame_no, num_frames_max, file, display=False, num_frames=1):
	"""
	extracts specific frame from video footage and displays it, if desired

	:param frame_no: integer value of desired frame
	:param num_frames_max: total number of frames within footage (could be extracted but to minimise the number of times
	                     the function is executed passed directly into the function
	:param file: video (as cap from OpenCV) from which the frame(s) is/are to be extracted
	:param display: display frame(s), if desired
	:param num_frames: number of frames to be extracted (1 by default to only return a single frame) If greater than 1,
	                  the PREVIOUS number of frames leading to the specified one will be extracted as well.

	:return: array of frames or single frame
	"""
	# print("Attempting to extract", num_frames, "images from file...")

	if frame_no - num_frames < 0:
		num_frames = frame_no
		# print("less than desired number of frames available! Extracting", num_frames, "instead!")

	all_frames = []

	if frame_no > num_frames_max:
		print("ERROR: frame number exceeds total number of frames in footage!")
		return

	for i in range(1, num_frames + 1):
		# skip back specified number of frames to include previous frames if desired / possible
		frame_val = frame_no - 1 + (i - num_frames)
		# set desired frame (Property identifier of cv2.CV_CAP_PROP_POS_FRAMES is 1, thus the first entry is 1
		file.set(1, frame_val)

		# Read the next frame from the video
		ret, frame = file.read()

		all_frames.append(frame)

		if display:
			# resize frame to fit screen
			new_height = int(np.shape(frame)[0] * 0.5)
			new_width = int(np.shape(frame)[1] * 0.5)
			frame = cv2.resize(frame, (new_width, new_height))

			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(
				frame,
				"frame: " + str(frame_no + (i - num_frames)),
				(int(new_width / 2) - 100, 35),
				font,
				0.8,
				(255, 255, 255),
				1,
				cv2.LINE_AA,
			)

			# Display the resulting frame
			cv2.imshow("Selected frames: ", frame)
			# Set wait key
			cv2.waitKey()

	# always reset frame from capture at the end to avoid incorrect skips during access
	file.set(1, 0)

	return all_frames


def extractPatches(frame_no, frames, tracks, patch_size=128, BW=True):
	"""
	extracts images patches for stacks and detections during TRAINING

	:param frame_no: desired frame of NEW detections
	:param frames: list of frames
	:param tracks: array of all labelled tracks imported
	:param patch_size: resolution (width / height in px) of extracted patches
	:param BW: Boolean, if True returns the patches in black and white

	:return: stacks of previous tracks and detections of new instances
	"""
	stacks = []
	stacks_label = []
	stacks_pos = []

	detections = []
	detections_label = []
	detections_pos = []

	# convert images to black and white if required
	if BW:
		for img in range(len(frames)):
			frames[img] = cv2.cvtColor(frames[img], cv2.COLOR_BGR2GRAY)
		blank_image = np.zeros((patch_size, patch_size), np.uint8)
	else:
		# coloured images require 3 channels
		blank_image = np.zeros((patch_size, patch_size, 3), np.uint8)

	# by default no images should be blank, exception occurs at the beginning of the footage when there are only
	# detections and no initialized tracks yet.
	num_empty_img = 0

	# insert blank images to fill stacks which would otherwise be too small
	blank_stack = []

	if frame_no - len(frames) < 0:
		num_empty_img = len(frames) - frame_no
		for img in range(num_empty_img):
			blank_stack.append(blank_image)

	# iterate over all available tracks, step size of two, as X and Y are given
	for track in range(1, tracks.shape[1], 2):
		stack = []
		pos = []
		no_detection = 0
		# iterate over all imported frames
		for img in range(len(frames) - num_empty_img):
			if tracks[frame_no + (img - len(frames) + num_empty_img), track] != 0:
				# the tracks are read as centres
				target_centre = np.asarray(
					[
						tracks[frame_no + (img - len(frames) + num_empty_img), track],
						tracks[
							frame_no + (img - len(frames) + num_empty_img), track + 1
						],
					]
				)

				# invert y axis, to fit openCV convention ( lower left -> (x=0,y=0) )
				target_centre[1] = frames[0].shape[0] - target_centre[1]
				# define the starting and ending point of the bounding box rectangle, defined by "target_size"
				px_start = target_centre - np.asarray(
					[math.floor(patch_size / 2), math.floor(patch_size / 2)]
				)
				px_end = target_centre + np.asarray(
					[math.floor(patch_size / 2), math.floor(patch_size / 2)]
				)
				# extract the defined rectangle of the track from the frame and save to the stack
				stack.append(
					frames[img][px_start[1] : px_end[1], px_start[0] : px_end[0]]
				)

				# save the position of each patch within the stack
				pos.append(target_centre)
			else:
				# if not detection can be found insert a black image instead
				stack.append(blank_image)
				# in case of a blank image / no defined patch the position is set to 0,0
				pos.append((0, 0))
				no_detection += 1

		# only return stacks which are active (i.e. at least one detection)
		if no_detection != len(frames):
			# add stack label to identify the stack later on
			label = int(track / 2 + 0.5)
			# set the newest entry of the stack as a detection for training purposes, retaining the label
			if np.bitwise_xor(stack[-1], blank_image).any():
				detections.append(stack[-1])
				detections_label.append(label)
				detections_pos.append(pos[-1])

			# remove the last entry of the stack, to only have in represented as a new detection
			del stack[-1]
			del pos[-1]
			# only return stacks if they are not empty without the newest detection
			if no_detection + 1 != len(frames):
				stacks.append(stack)
				stacks_label.append(label)
				stacks_pos.append(pos)

	# convert all outputs to numpy arrays
	stacks = np.array(stacks)
	stacks_label = np.array(stacks_label)
	stacks_pos = np.array(stacks_pos)

	detections = np.array(detections)
	detections_label = np.array(detections_label)
	detections_pos = np.array(detections_pos)

	return (
		stacks,
		stacks_label,
		stacks_pos,
		detections,
		detections_label,
		detections_pos,
	)


def display_patches(stacks, stack_labels):
	# simple function to show extracted patches
	for i in range(len(stacks)):
		# Display the resulting frame
		cv2.imshow(
			"First frame of each active stack: " + str(stack_labels[i]), stacks[i][0]
		)
		# Set wait key
		cv2.waitKey(1)
		time.sleep(0.1)


def sortByDistance(detections, stack_pos, detections_pos, labels, verbose=False):
	all_dist = []
	# find last valid position (non-zero element) in active stack
	valid_positions = [i for i, element in enumerate(stack_pos[:, 0]) if element != 0]
	last_valid_pos = stack_pos[valid_positions[-1]]

	for i in range(len(detections)):
		# compute distance between all detections and a given active stack
		all_dist.append(
			math.sqrt(
				((last_valid_pos[0] - detections_pos[i, 0]) ** 2)
				+ ((last_valid_pos[1] - detections_pos[i, 1]) ** 2)
			)
		)

	# sort detections based on smallest distance to given stack (closest detection first)
	detections_sorted = [
		i for _, i in sorted(zip(all_dist, detections), key=lambda pair: pair[0])
	]
	labels_sorted = [
		i for _, i in sorted(zip(all_dist, labels), key=lambda pair: pair[0])
	]
	detections_pos_sorted = [
		i for _, i in sorted(zip(all_dist, detections_pos), key=lambda pair: pair[0])
	]
	all_dist.sort()

	if verbose:
		print("\nvalid positions:", valid_positions)
		print("last valid position:", last_valid_pos)
		print("input labels:", labels)
		print("Sorted by distance to stack:", labels_sorted)
		print("Ascending pixel distances:", all_dist)

	return detections_sorted, labels_sorted, detections_pos_sorted


if __name__ == "__main__":
	path = "export/"
	cap = cv2.VideoCapture("2019-06-28_19-24-05.mp4")

	numFramesMax = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

	# this function is used to display the tracks you imported
	tracks = import_tracks(path, numFramesMax)
	display_video(cap, tracks, show=(380, 800), scale=0.5)

	# Check how much time is required to extract patches from images -> ~250 patches per second
	begin_timer = time.time()

	"""
    The section below simply demonstrates how stacks can be extracted from the imported video footage and your tracks.
    You may use this structure to extract videos with individual ants in them for your action recognition. The export
    pipeline is not written here yet, but it's simple to implement: A stack contains nothing but arrays of pixel
    information which you can for example export as an avi-jpeg sequence. I recommend looking into a few compression
    options, e.g. mp4 with a H264 codec to cut down on the required disk space.
    """

	# define your frame of interest and stack size (i.e. where to begin extraction and for how many frames)
	# last extractable frame value is numFramesMax - stack_size
	testy_frame_no = 1887
	# extract frame and frames leading up to it
	stack_size = 5

	# retrieve set of frames from imported video, set as cap
	testy_frame = get_exact_frame(
		frame_no=testy_frame_no,
		num_frames_max=numFramesMax,
		file=cap,
		display=False,
		num_frames=stack_size + 1,
	)  # one additional frame as the current frame are detections

	# extract patches of active tracks as stacks and detections of most recent frame
	(
		testy_stacks,
		testy_labels,
		testy_pos,
		testy_detections,
		testy_detection_labels,
		testy_detections_pos,
	) = extractPatches(frame_no=testy_frame_no, frames=testy_frame, tracks=tracks)

	# only print shape and labels if there are currently active stacks
	if testy_stacks.size != 0:
		print("Shape of stacks:", testy_stacks.shape)
		print("Number of active stacks:", len(testy_labels))
		print(
			"Positions of (first) active stack images:\n", testy_pos[-1]
		)  # display last entry
		print("Active stack labels:", testy_labels, "\n")
	else:
		print("No active stacks!\n")

	# only print shape and labels if there are currently active stacks
	if testy_detections.size != 0:
		print("Shape of detections:", testy_detections.shape)
		print("Number of new detections:", len(testy_detection_labels))
		print(
			"Positions of (first) detection:", testy_detections_pos[-1]
		)  # display last entry
		print("Detection labels:", testy_detection_labels, "\n")
	else:
		print("No detections!\n")

	print(
		"Required time to extract patches from footage:", str(time.time() - begin_timer)
	)

	# display extracted patches, if desired (displays first frame of each stack only)
	display_patches(stacks=testy_stacks, stack_labels=testy_labels)

	# pick an active stack (here: randomly) and sort detections by their distance to it. (Likely not relevant to you)
	example_track = np.random.randint(0, len(testy_labels))
	print("Using active stack with label:", testy_labels[example_track])
	detections_sorted, labels_sorted, detections_pos_sorted = sortByDistance(
		detections=testy_detections,
		stack_pos=testy_pos[example_track],
		detections_pos=testy_detections_pos,
		labels=testy_detection_labels,
		verbose=True,
	)

	# release video cap at the end of the script and close all windows
	cap.release()
	cv2.destroyAllWindows()
