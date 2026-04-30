PIXI_NO_PROGRESS=1 AIC_ACT_POLICY_PATH=/home/user/aic/.models/aic_act_policy \
pixi run ros2 run aic_model aic_model --ros-args \
  -p use_sim_time:=true \
  -p policy:=aic_model.policies.RunACTOffline
