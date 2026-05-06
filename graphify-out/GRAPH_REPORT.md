# Graph Report - aic  (2026-05-06)

## Corpus Check
- 109 files · ~118,003 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1058 nodes · 1652 edges · 125 communities (89 shown, 36 thin omitted)
- Extraction: 94% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 90 edges (avg confidence: 0.72)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `813f6f71`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]
- [[_COMMUNITY_Community 73|Community 73]]
- [[_COMMUNITY_Community 74|Community 74]]
- [[_COMMUNITY_Community 75|Community 75]]
- [[_COMMUNITY_Community 76|Community 76]]
- [[_COMMUNITY_Community 77|Community 77]]
- [[_COMMUNITY_Community 78|Community 78]]
- [[_COMMUNITY_Community 79|Community 79]]
- [[_COMMUNITY_Community 80|Community 80]]
- [[_COMMUNITY_Community 81|Community 81]]
- [[_COMMUNITY_Community 82|Community 82]]
- [[_COMMUNITY_Community 83|Community 83]]
- [[_COMMUNITY_Community 84|Community 84]]
- [[_COMMUNITY_Community 85|Community 85]]
- [[_COMMUNITY_Community 86|Community 86]]
- [[_COMMUNITY_Community 90|Community 90]]
- [[_COMMUNITY_Community 92|Community 92]]

## God Nodes (most connected - your core abstractions)
1. `ProximityDataGenerator` - 40 edges
2. `AicModel` - 23 edges
3. `TrainingFrameSink` - 19 edges
4. `ComputeScore()` - 18 edges
5. `Policy` - 18 edges
6. `MoveRobotCallback` - 15 edges
7. `CreateAndCancelTaskNode` - 14 edges
8. `AICRobotAICController` - 13 edges
9. `update()` - 12 edges
10. `handle_trial()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `CheatCodeFixed` --uses--> `MoveRobotCallback`  [INFERRED]
  aic_example_policies/aic_example_policies/ros/CheatCodeFixed.py → aic_model/aic_model/policy.py
- `CheatCodeFixed` --uses--> `Policy`  [INFERRED]
  aic_example_policies/aic_example_policies/ros/CheatCodeFixed.py → aic_model/aic_model/policy.py
- `WaveArm` --uses--> `MoveRobotCallback`  [INFERRED]
  aic_example_policies/aic_example_policies/ros/WaveArm.py → aic_model/aic_model/policy.py
- `WaveArm` --uses--> `Policy`  [INFERRED]
  aic_example_policies/aic_example_policies/ros/WaveArm.py → aic_model/aic_model/policy.py
- `SpeedDemon` --uses--> `Policy`  [INFERRED]
  aic_example_policies/aic_example_policies/ros/SpeedDemon.py → aic_model/aic_model/policy.py

## Hyperedges (group relationships)
- **Policy-to-Controller Interface Stack** — policy_policy_integration_doc, aic_interfaces_ai_challenge_interfaces_doc, aic_controller_aic_controller_doc, aic_interfaces_observation_message, aic_interfaces_motionupdate_interface, aic_interfaces_jointmotionupdate_interface [INFERRED 0.87]
- **Qualification Evaluation Contract** — qualification_phase_technical_overview_doc, scoring_scoring_doc, challenge_rules_challenge_rules_doc, qualification_phase_three_trial_protocol, scoring_three_tier_scoring_model, challenge_rules_aic_model_lifecycle_contract [INFERRED 0.84]
- **Submission Packaging and Registry Pipeline** — getting_started_getting_started_doc, submission_submission_guidelines_doc, submission_steps_submission_steps_doc, custom_dockerfile_custom_dockerfile_doc, submission_ecr_submission_workflow [INFERRED 0.86]
- **Visual Proximity Control Flow** — task_conditioned_candidate_selector, tri_camera_keypoint_triangulation, visual_servo_state_machine, motionupdate_velocity_commands [EXTRACTED 0.90]
- **Training Generation Pipeline** — proximity_teacher_policy, split_generator_control_data_plane, frame_sink_sidecar_pipeline, organizer_reset_ordering [EXTRACTED 0.90]
- **Evaluation Orchestration Stack** — aic_gz_bringup_launch_env, aic_engine_orchestrator, tier2_scoring_topic_set, ros_gz_scoring_bridge_topics [INFERRED 0.82]
- **Eval/Model Service Mesh** — docker_compose_eval_service, docker_compose_model_service, docker_compose_zenoh_transport_config [EXTRACTED 0.95]
- **Controller Command Contract** — aic_controller_controller_library, aic_control_interfaces_package, aic_adapter_adapter_executable [INFERRED 0.82]
- **Simulation-to-Scoring Feedback Loop** — aic_gazebo_off_limit_contacts_plugin, aic_engine_scoring_topics_manifest, aic_scoring_scoring_library [INFERRED 0.76]
- **ROS 2 Package Build Configuration** — cmakelists_aic_assets_project, cmakelists_ament_cmake, cmakelists_ament_package [INFERRED 0.81]

## Communities (125 total, 36 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (23): main(), SceneMarkerPublisher, main(), TFTreeDumper, Node, HomeTrajectoryNode, main(), compare_trajectories() (+15 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (28): KeyboardEndEffectorTeleop, KeyboardEndEffectorTeleopConfig, KeyboardJointTeleop, KeyboardJointTeleopConfig, action_features(), AICRobotAICControllerConfig, AICRos2Interface, CameraImageScaling (+20 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (50): ACT-First Learned Policy Plan, add_cable_plugin Refinement Flow, AIC Controller Impedance Settings, AIC Engine Orchestrator, aic_gz_bringup Launch Environment, aic_model Policy Framework, aic_bringup CMakeLists, AIC ROS2 Controllers Config (+42 more)

### Community 3 - "Community 3"
Cohesion: 0.08
Nodes (9): _decode_image_from_meta(), EpisodeSpec, main(), _postprocess_episode_images(), ProximityDataGenerator, _rpy_to_quat(), _stamp_to_sec(), _task_to_dict() (+1 more)

### Community 4 - "Community 4"
Cohesion: 0.08
Nodes (46): Access Control, Zenoh ACL Enforcement, aic_controller Documentation, Controller Impedance Pipeline, AI Challenge Interfaces, InsertCable Action Interface, JointMotionUpdate Command Interface, MotionUpdate Command Interface (+38 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (29): ActionsCfg, AICTaskEnvCfg, AICTaskSceneCfg, CommandsCfg, EventCfg, ObservationsCfg, PolicyCfg, Command terms for the MDP. (+21 more)

### Community 6 - "Community 6"
Cohesion: 0.15
Nodes (35): CalculateInverseProportionalScore(), ComputeScore(), ComputeTier3Score(), ContactsCallback(), ControllerStateCallback(), deserialize_from_rosbag(), EndEffectorPose(), GetContactsScore() (+27 more)

### Community 7 - "Community 7"
Cohesion: 0.17
Nodes (33): activate_model_node(), calculate_total_score(), check_endpoints(), check_model(), cleanup_model_node(), configure_model_node(), deactivate_model_node(), Engine() (+25 more)

### Community 8 - "Community 8"
Cohesion: 0.14
Nodes (17): _camera_info_to_dict(), _decode_image_from_meta(), _generate_visibility_occlusion_labels(), _intrinsics_from_camera_info(), _label_unavailable(), main(), _point_from_transform(), _pose_to_dict() (+9 more)

### Community 9 - "Community 9"
Cohesion: 0.16
Nodes (23): clamp_joint_reference_to_limits(), clamp_reference_to_limits(), command_interface_configuration(), Controller(), interpolate_impedance_parameters(), on_activate(), on_cleanup(), on_configure() (+15 more)

### Community 10 - "Community 10"
Cohesion: 0.1
Nodes (22): contact_net_forces(), Net contact forces (world frame) from the contact sensor, flattened for policy o, body_lin_acc_l2(), ee_reaching_bonus(), joint_acc_l2(), joint_pos_limits(), joint_torques_l2(), orientation_command_error() (+14 more)

### Community 11 - "Community 11"
Cohesion: 0.11
Nodes (4): AicModel, main(), Set a motion target for the robot.          There are two ways to move the robot, LifecycleNode

### Community 12 - "Community 12"
Cohesion: 0.12
Nodes (5): AICRobotAICController, Robot, _img_to_tensor(), Convert ROS Observation message into dictionary of normalized tensors., RunACT

### Community 13 - "Community 13"
Cohesion: 0.26
Nodes (18): add_package_runtime_dependencies(), _append_unique_value(), get_commands(), get_packages(), handle_dsv_types_except_source(), _include_comments(), main(), order_packages() (+10 more)

### Community 15 - "Community 15"
Cohesion: 0.22
Nodes (4): AICCartesianTeleoperatorNode, main(), Callback for keyboard listener when a key is pressed., Callback for keyboard listener when a key is released.

### Community 16 - "Community 16"
Cohesion: 0.38
Nodes (10): Cleanup(), Configure(), findLinkInModel(), IsModelValid(), MakeStatic(), PostUpdate(), PreUpdate(), Reset() (+2 more)

### Community 17 - "Community 17"
Cohesion: 0.23
Nodes (4): AICTeleoperatorNode, main(), Callback for keyboard listener when a key is pressed., Callback for keyboard listener when a key is released.

### Community 18 - "Community 18"
Cohesion: 0.21
Nodes (12): Training Campaign Runbook, Lifecycle Model Validation, aic_engine README, Engine Trial State Machine, Frame Sink Sidecar Pipeline, Training Data Generator Plan, Isolated Training Runtime Policy, Lifecycle Retry and Timeout Guardrails (+4 more)

### Community 19 - "Community 19"
Cohesion: 0.18
Nodes (4): Policy, GentleGiant, Policy that moves the arm slowly and smoothly using low stiffness     and high d, WaveArm

### Community 20 - "Community 20"
Cohesion: 0.35
Nodes (4): _normalize(), ProximityTeacher, Training-only teacher that reaches no-contact standoff from target port., _xyz()

### Community 22 - "Community 22"
Cohesion: 0.35
Nodes (7): CartesianImpedanceAction(), CartesianImpedanceParameters(), compute(), compute_nullspace_torque(), compute_smooth_right_pseudo_inverse(), configure(), single_joint_avoidance_torque()

### Community 23 - "Community 23"
Cohesion: 0.47
Nodes (9): camera_index(), decode_ros_image(), _emit_progress(), find_episode_dirs(), main(), parse_args(), process_episode(), _progress() (+1 more)

### Community 24 - "Community 24"
Cohesion: 0.31
Nodes (4): Policy, Return the current time from the node's clock (sim-time aware)., Sleep for the given duration using the node's clock (sim-time aware)., Invoke the move_robot callback to request the supplied Pose.          This is a

### Community 25 - "Community 25"
Cohesion: 0.51
Nodes (8): convert_episode(), decode_ros_image(), episode_number(), list_episode_dirs(), main(), parse_args(), parse_episode_list(), should_convert_episode()

### Community 30 - "Community 30"
Cohesion: 0.29
Nodes (10): aic_adapter Executable, ControllerState Message, JointMotionUpdate Message, MotionUpdate Message, aic_control_interfaces Package, aic_controller Library, Scoring Topic Subscription Manifest, OffLimitContactsPlugin (+2 more)

### Community 31 - "Community 31"
Cohesion: 0.44
Nodes (7): Median(), ParseStats(), ScoringTier1(), Stats(), TopicCallback(), TopicStatsTier1(), Update()

### Community 33 - "Community 33"
Cohesion: 0.28
Nodes (3): CheatCode, Wait for a TF frame to become available., Find the gripper pose that results in plug alignment.

### Community 35 - "Community 35"
Cohesion: 0.28
Nodes (5): main(), Serve xacro expansion requests from the training bringup environment., Expand a xacro file in a package share directory into XML., Run the ROS 2 node until shutdown., XacroExpanderNode

### Community 36 - "Community 36"
Cohesion: 0.28
Nodes (9): AIC-Task-v0 Environment, Isaac Lab Demo Recording Workflow, Isaac Lab RL Training Workflow, Isaac Lab Teleoperation Workflow, LeRobot Official Documentation, LeRobot Recording Workflow, Target Mode Requirement Rationale, LeRobot Teleoperation Workflow (+1 more)

### Community 37 - "Community 37"
Cohesion: 0.22
Nodes (9): Robot URDF and World Assets, Trial Scene Definitions, publish_scene_markers.py, CablePlugin, WorldSdfGeneratorPlugin, aic_task 0.1.0 Initial Template Entry, Future SDF-to-USD Export Pipeline, AIC Isaac Lab Integration Workflow (+1 more)

### Community 38 - "Community 38"
Cohesion: 0.28
Nodes (9): Sample Scoring Configuration, Trial Task Definitions, InsertCable Action, aic_task_interfaces Package, Task Message, eval Service, Internal Docker Network, model Service (+1 more)

### Community 39 - "Community 39"
Cohesion: 0.32
Nodes (5): ABC, insert_cable(), MoveRobotCallback, Move the robot using either Cartesian or joint-space commands.      This functio, Protocol

### Community 40 - "Community 40"
Cohesion: 0.36
Nodes (4): main(), RateLimiter, Convenience class for enforcing rates in loops., Collect demonstrations from the environment using teleop interfaces.

### Community 41 - "Community 41"
Cohesion: 0.39
Nodes (6): compare_states(), main(), pause_cb(), play_cb(), Compare states from dataset and runtime.      Returns:         Tuple of (states_, Replay episodes loaded from a file.

### Community 42 - "Community 42"
Cohesion: 0.36
Nodes (6): add_rsl_rl_args(), parse_rsl_rl_cfg(), Add RSL-RL arguments to the parser.      Args:         parser: The parser to add, Parse configuration for RSL-RL agent based on inputs.      Args:         task_na, Update configuration for RSL-RL agent based on inputs.      Args:         agent_, update_rsl_rl_cfg()

### Community 43 - "Community 43"
Cohesion: 0.36
Nodes (4): compute(), configure(), JointImpedanceAction(), JointImpedanceParameters()

### Community 44 - "Community 44"
Cohesion: 0.25
Nodes (8): publish_low_bandwidth_previews.py, Episode Capture Services, aic_training_interfaces Package, bin_to_webp_converter.py, aic_training_utils Package, proximity_data_generator.py, training_frame_sink.py, xacro_expander.py

### Community 45 - "Community 45"
Cohesion: 0.62
Nodes (5): Configure(), CreateCollisionData(), InitializeOffLimitEntities(), ParseSDF(), PreUpdate()

### Community 46 - "Community 46"
Cohesion: 0.48
Nodes (5): eigen_to_wrench_msg(), exp_map_quaternion(), integrate_pose(), log_map_quaternion(), wrench_msg_to_eigen()

### Community 47 - "Community 47"
Cohesion: 0.38
Nodes (3): compute(), configure(), GravityCompensationAction()

### Community 48 - "Community 48"
Cohesion: 0.48
Nodes (5): main(), postprocess_robot_xml(), postprocess_world_xml(), Apply automated corrections to world XML (replaces manual edits)., Apply automated corrections to robot XML (replaces manual edits).

### Community 49 - "Community 49"
Cohesion: 0.48
Nodes (5): apply_post_processing_fixes(), convert_sdf_to_mjcf(), main(), Convert an SDF file to MuJoCo MJCF format using sdformat_mjcf.      Args:, Apply manual fixes to the generated MJCF files.      TODO: Automate common fixes

### Community 50 - "Community 50"
Cohesion: 0.57
Nodes (5): AicAdapterNode(), image_callback(), main(), ReorderJointArray(), ReorderJointState()

### Community 51 - "Community 51"
Cohesion: 0.29
Nodes (7): ScoringPlugin, aic_scoring Shared Library, scoring_tier1_main Executable, scoring_tier2_main Executable, Tiered Scoring Pipeline, Scoring Failure Messages, Scoring Result Report Template

### Community 52 - "Community 52"
Cohesion: 0.38
Nodes (7): aic_assets CMakeLists, aic_assets project, ament_cmake, ament_environment_hooks, ament_package, models directory, ${PROJECT_NAME}.dsv.in hook template

### Community 53 - "Community 53"
Cohesion: 0.53
Nodes (4): Configure(), PreUpdate(), Reset(), ~ResetJointsPlugin()

### Community 57 - "Community 57"
Cohesion: 0.33
Nodes (6): ChangeTargetMode Service, Clamp-to-Limits Safety Parameters, Cartesian Impedance Parameter Set, Joint Impedance Parameter Set, aic_controller_parameters Library, Controller Target Mode Parameter

### Community 58 - "Community 58"
Cohesion: 0.6
Nodes (3): Configure(), PreUpdate(), Reset()

### Community 59 - "Community 59"
Cohesion: 0.6
Nodes (3): generate_launch_description(), launch_setup(), on_aic_engine_exit()

### Community 60 - "Community 60"
Cohesion: 0.5
Nodes (3): generate_launch_description(), Training bringup for Gazebo-based AIC workflows., Launch the standard Gazebo bringup together with training-only tools.

### Community 61 - "Community 61"
Cohesion: 0.6
Nodes (3): launch_viewer(), main(), Load MuJoCo scene and launch interactive viewer.      Args:         scene_path:

### Community 82 - "Community 82"
Cohesion: 0.67
Nodes (3): aic_engine_interfaces Package, ResetJoints Service, ResetJointsPlugin

### Community 83 - "Community 83"
Cohesion: 1.0
Nodes (3): aic_model policy.py Interface, ProximityTeacher Policy, RunACTOffline Policy

## Ambiguous Edges - Review These
- `Three-Tier Scoring Model` → `Baseline Policy Outcome Expectations`  [AMBIGUOUS]
  docs/scoring_tests.md · relation: conceptually_related_to
- `Tier 2 Plug/Port Entity Pairs` → `Duplicate Tier 2 Entity Entries`  [AMBIGUOUS]
  aic_scoring/config/tier2.yaml · relation: conceptually_related_to
- `Trial Task Definitions` → `InsertCable Action`  [AMBIGUOUS]
  aic_engine/config/sample_config.yaml · relation: conceptually_related_to
- `proximity_data_generator.py` → `Episode Capture Services`  [AMBIGUOUS]
  aic_utils/aic_training_utils/CMakeLists.txt · relation: conceptually_related_to

## Knowledge Gaps
- **128 isolated node(s):** `Find packages based on colcon-specific files created during installation.      :`, `Check the path and if it exists extract the packages runtime dependencies.`, `Order packages topologically.      :param dict packages: A mapping from package`, `Reduce the set of packages to the ones part of the circular dependency.      :pa`, `CheatCode variant with corrected quaternion math and tighter runtime behavior.` (+123 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **36 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Three-Tier Scoring Model` and `Baseline Policy Outcome Expectations`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Tier 2 Plug/Port Entity Pairs` and `Duplicate Tier 2 Entity Entries`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `Trial Task Definitions` and `InsertCable Action`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **What is the exact relationship between `proximity_data_generator.py` and `Episode Capture Services`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `TrainingFrameSink` connect `Community 8` to `Community 0`?**
  _High betweenness centrality (0.019) - this node is a cross-community bridge._
- **Why does `AICRobotAICController` connect `Community 12` to `Community 1`?**
  _High betweenness centrality (0.017) - this node is a cross-community bridge._
- **Why does `ProximityDataGenerator` connect `Community 3` to `Community 0`?**
  _High betweenness centrality (0.014) - this node is a cross-community bridge._