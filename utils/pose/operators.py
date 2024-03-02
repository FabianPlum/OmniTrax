import bpy
from bpy.props import BoolProperty as BoolProperty
import numpy as np
import cv2
import os
import yaml


class OMNITRAX_OT_PoseEstimationOperator(bpy.types.Operator):
	"""
	Run Pose estimation on the tracked animals
	ESTIMATE POSES: Run pose estimation on tracked ROIs (defaulting to full frame, when no tracks are present)
	ESTIMATE POSES [full frame]: Run (single subject) pose estimation on full resolution original video footage
	"""

	bl_idname = "scene.pose_estimation_run"
	bl_label = "Run Pose Estimation (using tracks as inputs)"

	fullframe: BoolProperty(
		name="Inference on full frame",
		description="Run pose estimation inference on entire frame instead of individual tracks.",
		default=False,
	)

	def execute(self, context):
		# check the DLC input folder checks out
		model_path = bpy.path.abspath(context.scene.pose_network_path)
		if not os.path.isdir(model_path):
			self.report(
				{"ERROR"},
				"Enter a valid path to a trained and exported DLC model folder",
			)
			return {"CANCELLED"}

		print("\nRUNNING POSE ESTIMATION\n")

		# check if any tracks exist for the currently selected clip. If not, run full frame pose estimation
		if len(context.edit_movieclip.tracking.objects[0].tracks) == 0:
			print("INFO: Found no tracks for current clip.")
			self.fullframe = True

		if self.fullframe:
			print("INFO: Running inference on full frame!")

		global dlc_proc
		global dlc_live
		global network_initialised
		global pose_cfg
		global pose_joint_header_l1
		global pose_joint_header_l2
		global pose_joint_header_l3
		global pose_joint_names

		if "dlc_proc" not in globals():
			from dlclive import DLCLive, Processor

			try:
				dlc_proc = Processor()
				print("Loading DLC Network from", model_path)
				dlc_live = DLCLive(
					model_path, processor=dlc_proc, pcutoff=context.scene.pose_pcutoff
				)

				# create a list of join names from those defined in the pose_cfg.yaml file
				dlc_pose_cfg = os.path.join(model_path, "pose_cfg.yaml")
				with open(dlc_pose_cfg, "r") as stream:
					pose_cfg_yaml = yaml.safe_load(stream)

				pose_joint_names = pose_cfg_yaml["all_joints_names"]
				pose_joint_header_l1 = "scorer," + ",".join(
					"OmniTrax,OmniTrax,OmniTrax" for e in pose_joint_names
				)
				pose_joint_header_l2 = "bodyparts," + ",".join(
					str(e) + "," + str(e) + "," + str(e) for e in pose_joint_names
				)
				pose_joint_header_l3 = "coords," + ",".join(
					"x," + "y," + "likelihood" for e in pose_joint_names
				)

			except:
				print("Failed to load trained network... Check your model path!")
				# remove dlc processor to attempt to re-initialise the network after user applies corrections
				del dlc_proc
				return {"FINISHED"}
		else:
			print("Initialised DLC Network found!")

		# next, try to load skeleton relationships from the config file
		try:
			model_path = bpy.path.abspath(context.scene.pose_network_path)
			dlc_config_path = os.path.join(model_path, "config.yaml")
			with open(dlc_config_path, "r") as stream:
				config_yaml = yaml.safe_load(stream)
				print("skeleton configuration:\n", config_yaml["skeleton"])

				# now, match the skeleton elements to their IDs to draw them as overlays
				skeleton = []
				try:
					for bone in config_yaml["skeleton"]:
						skeleton.append(
							[
								pose_joint_names.index(bone[0]),
								pose_joint_names.index(bone[1]),
							]
						)

					print("skeleton links:\n", skeleton)
				except ValueError:
					print(
						"Your config skeleton and pose joint names do not match!"
						"\n could not create overlay skeleton!"
					)
					skeleton = []

		except FileNotFoundError:
			print(
				"No config.yaml file found!\n"
				"Place your config file in the exported model folder to overlay the skeleton!\n"
			)
			skeleton = []

		try:
			clip = context.edit_movieclip
			clip_path = bpy.path.abspath(bpy.context.edit_movieclip.filepath)

			clip_width = clip.size[0]
			clip_height = clip.size[1]
		except:
			print(
				"You need to load and track a video, before running pose estimation!\n"
			)
			return {"FINISHED"}

		first_frame = context.scene.frame_start
		last_frames = context.scene.frame_end

		# now we can load the captured video file and display it
		cap = cv2.VideoCapture(clip_path)
		fps = cap.get(cv2.CAP_PROP_FPS)
		video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

		if "network_initialised" not in globals():
			network_initialised = False

		ROI_size = int(context.scene.pose_constant_size / 2)

		print("Using", clip_path, "for pose estimation...")

		if self.fullframe:
			# run full frame pose-inference
			if context.scene.pose_save_video:
				video_output = (
					bpy.path.abspath(bpy.context.edit_movieclip.filepath)[:-4]
					+ "_POSE_fullframe.mp4"
				)
				video_out = cv2.VideoWriter(
					video_output,
					cv2.VideoWriter_fourcc(*"mp4v"),
					fps,
					(int(video_width), int(video_height)),
				)
			track_pose = {}

			for frame_id in range(first_frame, last_frames):
				try:
					cap.set(1, frame_id)
					ret, frame_temp = cap.read()

					if ret:
						dlc_input_img = frame_temp

						# initialise network (if it has not been initialised yet)
						if not network_initialised:
							dlc_live.init_inference(dlc_input_img)
							network_initialised = True

						# estimate pose in cropped frame
						pose = dlc_live.get_pose(dlc_input_img)
						thresh = context.scene.pose_pcutoff

						track_pose[str(frame_id)] = pose.flatten()

						for p, point in enumerate(pose):
							if point[2] >= thresh:
								dlc_input_img = cv2.circle(
									dlc_input_img,
									(int(point[0]), int(point[1])),
									context.scene.pose_point_size,
									(
										int(255 * p / len(pose_joint_names)),
										int(255 - 255 * p / len(pose_joint_names)),
										200,
									),
									-1,
								)

								if context.scene.pose_show_labels:
									dlc_input_img = cv2.putText(
										dlc_input_img,
										pose_joint_names[p],
										(int(point[0]), int(point[1])),
										cv2.FONT_HERSHEY_SIMPLEX,
										1,
										(
											int(255 * p / len(pose_joint_names)),
											int(255 - 255 * p / len(pose_joint_names)),
											200,
										),
										1,
									)

						for b, bone in enumerate(skeleton):
							if (
								pose[bone[0]][2] >= thresh
								and pose[bone[1]][2] >= thresh
							):
								if context.scene.pose_plot_skeleton:
									dlc_input_img = cv2.line(
										dlc_input_img,
										(int(pose[bone[0]][0]), int(pose[bone[0]][1])),
										(int(pose[bone[1]][0]), int(pose[bone[1]][1])),
										(120, 220, 120),
										context.scene.pose_skeleton_bone_width,
									)

						cv2.imshow("DLC Pose Estimation", dlc_input_img)
						if cv2.waitKey(1) & 0xFF == ord("q"):
							break
						if context.scene.pose_save_video:
							video_out.write(dlc_input_img)

				except Exception as e:
					print(e)
					track_pose[str(frame_id)] = np.array([0])

			if context.scene.pose_export_pose:
				pose_output_file = open(
					bpy.path.abspath(bpy.context.edit_movieclip.filepath)[:-4]
					+ "_POSE_fullframe.csv",
					"w",
				)
				# write header line
				pose_output_file.write(pose_joint_header_l1 + "\n")
				pose_output_file.write(pose_joint_header_l2 + "\n")
				pose_output_file.write(pose_joint_header_l3 + "\n")

				for key, value in track_pose.items():
					line = key + "," + ",".join(str(e) for e in value.flatten())
					pose_output_file.write(line + "\n")
				pose_output_file.close()

		else:
			# run per-track pose-inference
			for track in clip.tracking.objects[0].tracks:
				if context.scene.pose_save_video:
					video_output = (
						bpy.path.abspath(bpy.context.edit_movieclip.filepath)[:-4]
						+ "_POSE_"
						+ track.name
						+ ".mp4"
					)
					video_out = cv2.VideoWriter(
						video_output,
						cv2.VideoWriter_fourcc(*"mp4v"),
						fps,
						(
							int(context.scene.pose_constant_size),
							int(context.scene.pose_constant_size),
						),
					)

				# keeps track of the pose info at each frame for each track
				# frame_id : [[joint_a_X, joint_a_Y, joint_a_Confidence],
				#             [joint_b_X, joint_b_Y, joint_b_Confidence],  ...]
				track_pose = {}

				for frame_id in range(first_frame, last_frames):
					marker = track.markers.find_frame(frame_id)
					try:
						if marker:
							marker_x = round(marker.co.x * clip_width)
							marker_y = round(marker.co.y * clip_height)
							print(
								"Frame:",
								frame_id,
								" : ",
								"X",
								marker_x,
								",",
								"Y",
								marker_y,
							)

							cap.set(1, frame_id)
							ret, frame_temp = cap.read()
							if ret:
								# first, create an empty image object to be filled with the ROI
								# this is important in case the detection lies close to the edge
								# where the ROI would go outside the image
								dlc_input_img = np.zeros(
									[ROI_size * 2, ROI_size * 2, 3], dtype=np.uint8
								)
								dlc_input_img.fill(0)  # fill with zeros

								if context.scene.pose_enforce_constant_size:
									true_min_x = marker_x - ROI_size
									true_max_x = marker_x + ROI_size
									true_min_y = clip_height - marker_y - ROI_size
									true_max_y = clip_height - marker_y + ROI_size

									min_x = max([0, true_min_x])
									max_x = min([clip.size[0], true_max_x])
									min_y = max([0, true_min_y])
									max_y = min([clip.size[1], true_max_y])
									# crop frame to detection and rescale
									frame_cropped = frame_temp[min_y:max_y, min_x:max_x]

									# place the cropped frame in the previously created empty image
									x_min_offset = max([0, -true_min_x])
									x_max_offset = min(
										[
											ROI_size * 2,
											ROI_size * 2 - (true_max_x - clip.size[0]),
										]
									)
									y_min_offset = max([0, -true_min_y])
									y_max_offset = min(
										[
											ROI_size * 2,
											ROI_size * 2 - (true_max_y - clip.size[1]),
										]
									)

									print(
										"Cropped image ROI:",
										x_min_offset,
										x_max_offset,
										y_min_offset,
										y_max_offset,
									)
									dlc_input_img[
										y_min_offset:y_max_offset,
										x_min_offset:x_max_offset,
									] = frame_cropped
								else:
									bbox = marker.pattern_corners
									true_min_x = marker_x + int(bbox[0][0] * clip_width)
									true_max_x = marker_x - int(bbox[0][0] * clip_width)
									true_min_y = (
										clip_height
										- marker_y
										- int(bbox[0][1] * clip_height)
									)
									true_max_y = (
										clip_height
										- marker_y
										+ int(bbox[0][1] * clip_height)
									)
									true_width = true_max_x - true_min_x
									true_height = true_max_y - true_min_y

									if true_height < 0:  # flip y axis, if required
										true_min_y, true_max_y = true_max_y, true_min_y
										true_height = -true_height

									if true_width < 0:  # flip x axis, if required
										true_min_x, true_max_x = true_max_x, true_min_x
										true_width = -true_width

									print(
										"Cropped image ROI:",
										true_min_x,
										true_max_x,
										true_min_y,
										true_max_y,
										"\n Detection h/w:",
										true_height,
										true_width,
									)

									# resize image and maintain aspect ratio to the specified ROI
									if true_width >= true_height:
										rescale_width = int(ROI_size * 2)
										rescale_height = int(
											(true_height / true_width) * ROI_size * 2
										)
										border_height = max(
											[
												int(
													(rescale_width - rescale_height) / 2
												),
												0,
											]
										)
										print(
											rescale_width, rescale_height, border_height
										)
										frame_cropped = cv2.resize(
											frame_temp[
												true_min_y:true_max_y,
												true_min_x:true_max_x,
											],
											(rescale_width, rescale_height),
										)

										dlc_input_img[
											border_height : rescale_height
											+ border_height,
											:,
										] = frame_cropped
									else:
										rescale_width = int(
											(true_width / true_height) * ROI_size * 2
										)
										rescale_height = int(ROI_size * 2)
										border_width = max(
											[
												int(
													abs(
														(rescale_height - rescale_width)
													)
													/ 2
												),
												0,
											]
										)
										frame_cropped = cv2.resize(
											frame_temp[
												true_min_y:true_max_y,
												true_min_x:true_max_x,
											],
											(rescale_width, rescale_height),
										)

										dlc_input_img[
											:,
											border_width : rescale_width + border_width,
										] = frame_cropped

								# initialise network (if it has not been initialised yet)
								if not network_initialised:
									dlc_live.init_inference(dlc_input_img)
									network_initialised = True

								# estimate pose in cropped frame
								pose = dlc_live.get_pose(dlc_input_img)
								thresh = context.scene.pose_pcutoff

								track_pose[str(frame_id)] = pose.flatten()

								for p, point in enumerate(pose):
									if point[2] >= thresh:
										dlc_input_img = cv2.circle(
											dlc_input_img,
											(int(point[0]), int(point[1])),
											context.scene.pose_point_size,
											(
												int(255 * p / len(pose_joint_names)),
												int(
													255
													- 255 * p / len(pose_joint_names)
												),
												200,
											),
											-1,
										)

										if context.scene.pose_show_labels:
											dlc_input_img = cv2.putText(
												dlc_input_img,
												pose_joint_names[p],
												(int(point[0]), int(point[1])),
												cv2.FONT_HERSHEY_SIMPLEX,
												1,
												(
													int(
														255 * p / len(pose_joint_names)
													),
													int(
														255
														- 255
														* p
														/ len(pose_joint_names)
													),
													200,
												),
												1,
											)

								# TODO add robust joint angle calculation (for various species)
								"""
                                joint_angles = np.empty(42)
                                joint_angles_conf = np.empty(42)  # report confidence on each joint angle
                                main_body_axis = [pose[0][0] - pose[6][0], pose[0][1] - pose[6][1]]  # b_t to b_a
                                unit_vector_body_axis = main_body_axis / np.linalg.norm(main_body_axis)
                                for b, bone in enumerate(skeleton):
                                    if pose[bone[0]][2] >= thresh and pose[bone[1]][2] >= thresh:
                                        if context.scene.pose_export_pose:
                                            # save angles between keypoints
                                            if b < 42:
                                                bone_vector = [pose[bone[0]][0] - pose[bone[1]][0],
                                                               pose[bone[0]][1] - pose[bone[1]][1]]
                                                unit_vector_bone_vector = bone_vector / np.linalg.norm(bone_vector)
                                                dot_product = np.dot(unit_vector_body_axis, unit_vector_bone_vector)
                                                joint_angles[b] = np.arccos(np.clip(dot_product, -1.0, 1.0))
                                                joint_angles_conf[b] = pose[bone[0]][2] + pose[bone[1]][2]

                                        if context.scene.pose_plot_skeleton:
                                            dlc_input_img = cv2.line(dlc_input_img,
                                                                     (int(pose[bone[0]][0]), int(pose[bone[0]][1])),
                                                                     (int(pose[bone[1]][0]), int(pose[bone[1]][1])),
                                                                     (120, 220, 120),
                                                                     context.scene.pose_skeleton_bone_width)

                                # now get the angle of each leg by taking the median angle from each associated joint
                                leg_angles = np.array([np.average(joint_angles[1:3], weights=joint_angles_conf[1:3]),
                                                       np.average(joint_angles[8:10], weights=joint_angles_conf[8:10]),
                                                       np.average(joint_angles[15:17], weights=joint_angles_conf[15:17]),
                                                       np.average(joint_angles[22:24], weights=joint_angles_conf[22:24]),
                                                       np.average(joint_angles[29:31], weights=joint_angles_conf[29:31]),
                                                       np.average(joint_angles[36:38], weights=joint_angles_conf[36:38])])

                                track_pose[str(frame_id)] = np.concatenate((track_pose[str(frame_id)], leg_angles))
                                """
								for b, bone in enumerate(skeleton):
									if (
										pose[bone[0]][2] >= thresh
										and pose[bone[1]][2] >= thresh
									):
										if context.scene.pose_plot_skeleton:
											dlc_input_img = cv2.line(
												dlc_input_img,
												(
													int(pose[bone[0]][0]),
													int(pose[bone[0]][1]),
												),
												(
													int(pose[bone[1]][0]),
													int(pose[bone[1]][1]),
												),
												(120, 220, 120),
												context.scene.pose_skeleton_bone_width,
											)

								cv2.imshow("DLC Pose Estimation", dlc_input_img)
								if cv2.waitKey(1) & 0xFF == ord("q"):
									break
								if context.scene.pose_save_video:
									video_out.write(dlc_input_img)

					except Exception as e:
						print(e)
						track_pose[str(frame_id)] = np.array([0])

				if context.scene.pose_export_pose:
					pose_output_file = open(
						bpy.path.abspath(bpy.context.edit_movieclip.filepath)[:-4]
						+ "_POSE_"
						+ track.name
						+ ".csv",
						"w",
					)
					# write header line
					# pose_output_file.write("frame," + pose_joint_header + ",r1_deg,r2_deg,r3_deg,l1_deg,l2_deg,l3_deg\n")
					# TODO add robust joint angle calculation (see above)
					"""
                    replicate DLC prediction output file structure:
                    scorer    | OmniTrax  | OmniTrax  | OmniTrax   | ...
                    bodyparts | part_A    | part_A    | part_A     | ...
                    coords    | x         | y         | likelihood | ...
                    """
					# write header line
					pose_output_file.write(pose_joint_header_l1 + "\n")
					pose_output_file.write(pose_joint_header_l2 + "\n")
					pose_output_file.write(pose_joint_header_l3 + "\n")

					for key, value in track_pose.items():
						line = key + "," + ",".join(str(e) for e in value.flatten())
						pose_output_file.write(line + "\n")
					pose_output_file.close()

				print("\n")

		cv2.destroyAllWindows()

		# always reset frame from capture at the end to avoid incorrect skips during access
		cap.set(1, context.scene.frame_start - 1)
		cap.release()
		if context.scene.pose_save_video:
			video_out.release()
		print("Read all frames")

		return {"FINISHED"}


def register():
	bpy.utils.register_class(OMNITRAX_OT_PoseEstimationOperator)


def unregister():
	bpy.utils.unregister_class(OMNITRAX_OT_PoseEstimationOperator)
