# Configure the node (this loads the WaveArm policy)
pixi run ros2 lifecycle set /aic_model configure

# Activate the node
pixi run ros2 lifecycle set /aic_model activate

pixi run ros2 action send_goal /insert_cable aic_task_interfaces/action/InsertCable "{task: {id: 'test_run'}}"
