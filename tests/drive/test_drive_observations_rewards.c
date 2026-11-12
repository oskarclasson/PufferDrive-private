#include "include/drive_fixture.h"
#include "include/test.h"

static int test_observation_size_formula(void) {
    Drive env = {0};
    env.num_goals = 3;
    env.reward_conditioning = 0;
    env.obs_slots_partners_n = 2;
    env.obs_slots_lane_kept = 5;
    env.obs_slots_boundary_kept = 7;
    env.obs_slots_traffic_controls_n = 4;
    int expected = EGO_FEATURES + 3 * GOAL_FEATURES + 2 * PARTNER_FEATURES + 5 * LANE_FEATURES + 7 * BOUNDARY_FEATURES
        + 4 * TRAFFIC_CONTROL_FEATURES + OBS_VALID_COUNT_FEATURES;
    EXPECT_EQ_INT(compute_observation_size(&env), expected);

    env.reward_conditioning = 1;
    expected = EGO_FEATURES + NUM_REWARD_COEFS + 3 * GOAL_FEATURES + 2 * PARTNER_FEATURES + 5 * LANE_FEATURES
        + 7 * BOUNDARY_FEATURES + 4 * TRAFFIC_CONTROL_FEATURES + OBS_VALID_COUNT_FEATURES;
    EXPECT_EQ_INT(compute_observation_size(&env), expected);

    env.obs_partner_relative_velocity = 1;
    expected += 2 * PARTNER_RELATIVE_VELOCITY_FEATURES;
    EXPECT_EQ_INT(compute_observation_size(&env), expected);

    env.obs_lane_speed_limit = 1;
    expected += 5 * LANE_SPEED_LIMIT_FEATURES;
    EXPECT_EQ_INT(compute_observation_size(&env), expected);
    return 0;
}

static int check_lane_speed_limit_column(int lane_slots_kept, int boundary_slots_kept) {
    const float forced_limit_mps = 12.5f;
    srand(21);
    Drive plain = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 4, 0);
    plain.obs_slots_lane_kept = lane_slots_kept;
    plain.obs_slots_boundary_kept = boundary_slots_kept;
    allocate(&plain);
    c_reset(&plain);
    srand(21);
    Drive with_limit = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 4, 0);
    with_limit.obs_slots_lane_kept = lane_slots_kept;
    with_limit.obs_slots_boundary_kept = boundary_slots_kept;
    with_limit.obs_lane_speed_limit = 1;
    allocate(&with_limit);
    c_reset(&with_limit);

    EXPECT_EQ_INT(with_limit.active_agent_count, plain.active_agent_count);
    int plain_size = compute_observation_size(&plain);
    int limit_size = compute_observation_size(&with_limit);
    EXPECT_EQ_INT(limit_size - plain_size, plain.obs_slots_lane_kept * LANE_SPEED_LIMIT_FEATURES);
    int lane_base = EGO_FEATURES + plain.num_goals * GOAL_FEATURES + plain.obs_slots_partners_n * PARTNER_FEATURES;
    int limit_row_features = LANE_FEATURES + LANE_SPEED_LIMIT_FEATURES;
    int plain_tail = lane_base + plain.obs_slots_lane_kept * LANE_FEATURES;
    int limit_tail = lane_base + plain.obs_slots_lane_kept * limit_row_features;

    for (int agent_idx = 0; agent_idx < plain.active_agent_count; agent_idx++) {
        const float *plain_obs = &plain.observations[agent_idx * plain_size];
        const float *limit_obs = &with_limit.observations[agent_idx * limit_size];
        int lane_count = (int) limit_obs[limit_size - OBS_VALID_COUNT_FEATURES];
        EXPECT_TRUE(lane_count > 0);
        EXPECT_EQ_INT(memcmp(plain_obs, limit_obs, lane_base * sizeof(float)), 0);
        EXPECT_EQ_INT(
            memcmp(&plain_obs[plain_tail], &limit_obs[limit_tail], (plain_size - plain_tail) * sizeof(float)),
            0);
        for (int row = 0; row < plain.obs_slots_lane_kept; row++) {
            const float *plain_row = &plain_obs[lane_base + row * LANE_FEATURES];
            const float *limit_row = &limit_obs[lane_base + row * limit_row_features];
            EXPECT_EQ_INT(memcmp(plain_row, limit_row, LANE_FEATURES * sizeof(float)), 0);
            EXPECT_TRUE(row < lane_count ? limit_row[LANE_FEATURES] > 0.0f : limit_row[LANE_FEATURES] == 0.0f);
        }
    }

    for (int road_idx = 0; road_idx < with_limit.num_road_elements; road_idx++) {
        with_limit.lane_speed_limit_mps[road_idx] = forced_limit_mps;
    }
    compute_observations(&with_limit);
    for (int agent_idx = 0; agent_idx < with_limit.active_agent_count; agent_idx++) {
        const float *limit_obs = &with_limit.observations[agent_idx * limit_size];
        int lane_count = (int) limit_obs[limit_size - OBS_VALID_COUNT_FEATURES];
        for (int row = 0; row < with_limit.obs_slots_lane_kept; row++) {
            float expected = row < lane_count ? forced_limit_mps / with_limit.obs_norm_speed_mps : 0.0f;
            EXPECT_NEAR(limit_obs[lane_base + row * limit_row_features + LANE_FEATURES], expected, 1e-6f);
        }
    }

    free_allocated(&plain);
    free_allocated(&with_limit);
    return 0;
}

static int test_lane_speed_limit_column(void) {
    EXPECT_EQ_INT(check_lane_speed_limit_column(32, 32), 0);
    // fewer kept than observed slots enables road dropout (buffered + subsampled rows)
    EXPECT_EQ_INT(check_lane_speed_limit_column(12, 16), 0);
    return 0;
}

static int test_observation_zero_fill_and_valid_counts(void) {
    srand(3);
    Drive env = drive_test_env_config(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    env.obs_slots_partners_n = 4;
    env.obs_slots_lane_n = 8;
    env.obs_slots_boundary_n = 8;
    env.obs_slots_lane_kept = 4;
    env.obs_slots_boundary_kept = 4;
    env.obs_slots_traffic_controls_n = 2;
    allocate(&env);
    c_reset(&env);

    EXPECT_EQ_INT(env.active_agent_count, 1);
    int obs_size = compute_observation_size(&env);
    float *obs = env.observations;
    int partner_base = EGO_FEATURES + env.num_goals * GOAL_FEATURES;
    int road_base = partner_base + env.obs_slots_partners_n * PARTNER_FEATURES;
    int traffic_base
        = road_base + env.obs_slots_lane_kept * LANE_FEATURES + env.obs_slots_boundary_kept * BOUNDARY_FEATURES;
    int valid_base = obs_size - OBS_VALID_COUNT_FEATURES;

    EXPECT_NEAR(obs[valid_base + 2], 0.0f, 1e-5f);
    EXPECT_TRUE(obs[valid_base] >= 0.0f && obs[valid_base] <= (float) env.obs_slots_lane_kept);
    EXPECT_TRUE(obs[valid_base + 1] >= 0.0f && obs[valid_base + 1] <= (float) env.obs_slots_boundary_kept);
    EXPECT_TRUE(obs[valid_base + 3] >= 0.0f && obs[valid_base + 3] <= (float) env.obs_slots_traffic_controls_n);

    for (int i = 0; i < env.obs_slots_partners_n * PARTNER_FEATURES; i++) {
        EXPECT_NEAR(obs[partner_base + i], 0.0f, 1e-6f);
    }
    for (int i = traffic_base + (int) obs[valid_base + 3] * TRAFFIC_CONTROL_FEATURES; i < valid_base; i++) {
        EXPECT_NEAR(obs[i], 0.0f, 1e-6f);
    }

    free_allocated(&env);
    return 0;
}

static void init_reward_env(Drive *env, Agent *agent, Log *log, int *active, float *reward) {
    memset(env, 0, sizeof(*env));
    memset(agent, 0, sizeof(*agent));
    memset(log, 0, sizeof(*log));
    *agent = drive_test_agent(0.0f, 0.0f, 0.0f);
    active[0] = 0;
    env->agents = agent;
    env->active_agent_indices = active;
    env->logs = log;
    env->rewards = reward;
    env->active_agent_count = 1;
    env->dt = 0.1f;
    env->reward_goal = 2.0f;
    env->simulation_mode = SIMULATION_MODE_GIGAFLOW;
    env->num_goals = 3;
    agent->goal_count = 3;
    env->compute_eval_metrics = 0;
    agent->reward_coefs[REWARD_COEF_COLLISION] = 3.0f;
    agent->reward_coefs[REWARD_COEF_OFFROAD] = 4.0f;
    agent->reward_coefs[REWARD_COEF_STOP_LINE] = 5.0f;
    agent->metrics_array[LANE_ANGLE_IDX] = 1.0f;
}

static int test_reward_terminal_components(void) {
    Drive env;
    Agent agent;
    Log log;
    int active[1];
    float reward[1] = {0};

    init_reward_env(&env, &agent, &log, active, reward);
    agent.sim_speed = 10.0f;
    agent.metrics_array[COLLISION_IDX] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(env.rewards[0], -4.0f, 1e-5f);
    EXPECT_NEAR(log.reward_collision, -4.0f, 1e-5f);

    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.metrics_array[OFFROAD_IDX] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(env.rewards[0], -4.0f, 1e-5f);
    EXPECT_NEAR(log.reward_offroad, -4.0f, 1e-5f);

    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.metrics_array[RED_LIGHT_IDX] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(env.rewards[0], -5.0f, 1e-5f);
    EXPECT_NEAR(log.reward_red_light, -5.0f, 1e-5f);

    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.metrics_array[STOP_SIGN_IDX] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(env.rewards[0], -5.0f, 1e-5f);
    EXPECT_NEAR(log.reward_stop_sign, -5.0f, 1e-5f);
    EXPECT_NEAR(log.stop_sign_violation_rate, 1.0f, 1e-5f);

    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.metrics_array[REACHED_GOAL_IDX] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(env.rewards[0], 2.0f, 1e-5f);
    EXPECT_NEAR(log.reward_goal, 2.0f, 1e-5f);
    return 0;
}

static int test_reward_goal_speed_gating(void) {
    Drive env;
    Agent agent;
    Log log;
    int active[1];
    float reward[1] = {0};

    // GIGAFLOW: final waypoint reached above goal-speed → goal reward gated to 0
    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.metrics_array[REACHED_GOAL_IDX] = 1.0f;
    agent.current_goal_idx = agent.goal_count;
    agent.reward_coefs[REWARD_COEF_GOAL_SPEED] = 3.0f;
    agent.sim_speed = 10.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(log.reward_goal, 0.0f, 1e-5f);

    // Same final waypoint but below goal-speed → full goal reward
    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.metrics_array[REACHED_GOAL_IDX] = 1.0f;
    agent.current_goal_idx = agent.goal_count;
    agent.reward_coefs[REWARD_COEF_GOAL_SPEED] = 3.0f;
    agent.sim_speed = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(log.reward_goal, 2.0f, 1e-5f);
    return 0;
}

static int test_reward_coef_goal_speed_pinned(void) {
    Drive env = {0};
    Agent agent = {0};
    env.reward_randomization = 1;
    env.goal_speed = 3.0f;
    rng_seed(&env.rng_state, 7);

    env.goal_speed_randomization = 0;
    for (int sample_idx = 0; sample_idx < 16; sample_idx++) {
        generate_reward_coefs(&env, &agent);
        EXPECT_NEAR(agent.reward_coefs[REWARD_COEF_GOAL_SPEED], 3.0f, 1e-6f);
    }

    env.goal_speed_randomization = 1;
    int differs_from_pinned = 0;
    for (int sample_idx = 0; sample_idx < 16; sample_idx++) {
        generate_reward_coefs(&env, &agent);
        float goal_speed = agent.reward_coefs[REWARD_COEF_GOAL_SPEED];
        EXPECT_TRUE(goal_speed >= 0.0f && goal_speed <= 20.0f);
        differs_from_pinned += fabsf(goal_speed - 3.0f) > 1e-3f;
    }
    EXPECT_TRUE(differs_from_pinned > 0);
    return 0;
}

static int test_partner_obs_relative_velocity(void) {
    // Ego heading +x at 10 m/s; partner 20 m ahead heading +y at 5 m/s -> ego-frame relative velocity (-10, 5).
    Drive env = {0};
    env.obs_norm_speed_mps = 10.0f;
    env.obs_norm_xy_offset_m = 100.0f;
    env.obs_norm_z_m = 10.0f;
    env.obs_norm_veh_length_m = 10.0f;
    env.obs_norm_veh_width_m = 5.0f;
    env.obs_range_partner_m = 100.0f;
    env.obs_slots_partners_n = 2;
    env.num_agents = 2;
    env.active_agent_count = 2;
    int active[2] = {0, 1};
    env.active_agent_indices = active;
    Agent agents[2];
    agents[0] = drive_test_agent(0.0f, 0.0f, 0.0f);
    agents[0].sim_vx = 10.0f;
    agents[1] = drive_test_agent(20.0f, 0.0f, (float) M_PI / 2.0f);
    agents[1].sim_vy = 5.0f;
    update_agent_speed(&agents[0]);
    update_agent_speed(&agents[1]);
    env.agents = agents;
    PartnerScanEntry partner_scan[2] = {
        {agents[0].sim_x, agents[0].sim_y, agents[0].sim_z, 0},
        {agents[1].sim_x, agents[1].sim_y, agents[1].sim_z, 1},
    };
    float obs[2 * (PARTNER_FEATURES + PARTNER_RELATIVE_VELOCITY_FEATURES)];
    int partner_count = 0;

    env.obs_partner_relative_velocity = 0;
    memset(obs, 0, sizeof(obs));
    int end_idx = write_partner_obs(&env, &agents[0], 0, partner_scan, obs, 0, &partner_count);
    EXPECT_EQ_INT(partner_count, 1);
    EXPECT_EQ_INT(end_idx, 2 * PARTNER_FEATURES);
    EXPECT_NEAR(obs[7], 0.5f, 1e-5f);

    env.obs_partner_relative_velocity = 1;
    memset(obs, 0, sizeof(obs));
    end_idx = write_partner_obs(&env, &agents[0], 0, partner_scan, obs, 0, &partner_count);
    EXPECT_EQ_INT(end_idx, 2 * (PARTNER_FEATURES + PARTNER_RELATIVE_VELOCITY_FEATURES));
    EXPECT_NEAR(obs[7], 0.5f, 1e-5f);
    EXPECT_NEAR(obs[PARTNER_FEATURES], -1.0f, 1e-5f);
    EXPECT_NEAR(obs[PARTNER_FEATURES + 1], 0.5f, 1e-5f);
    return 0;
}

static int test_partner_obs_split_long_partner(void) {
    // Ego heading +x; 4 m car 10 m behind; 13 m truck 20 m ahead heading +y -> split at 7 m into 6.5 m pieces at y
    // -/+3.25.
    Drive env = {0};
    env.obs_norm_speed_mps = 10.0f;
    env.obs_norm_xy_offset_m = 100.0f;
    env.obs_norm_z_m = 10.0f;
    env.obs_norm_veh_length_m = 10.0f;
    env.obs_norm_veh_width_m = 5.0f;
    env.obs_range_partner_m = 100.0f;
    env.obs_partner_relative_velocity = 1;
    env.num_agents = 3;
    env.active_agent_count = 3;
    int active[3] = {0, 1, 2};
    env.active_agent_indices = active;
    Agent agents[3];
    agents[0] = drive_test_agent(0.0f, 0.0f, 0.0f);
    agents[1] = drive_test_agent(20.0f, 0.0f, (float) M_PI / 2.0f);
    agents[1].sim_length = 13.0f;
    agents[1].sim_vy = 5.0f;
    agents[2] = drive_test_agent(-10.0f, 0.0f, 0.0f);
    for (int agent_idx = 0; agent_idx < 3; agent_idx++) {
        update_agent_speed(&agents[agent_idx]);
    }
    env.agents = agents;
    PartnerScanEntry partner_scan[3] = {
        {agents[0].sim_x, agents[0].sim_y, agents[0].sim_z, 0},
        {agents[1].sim_x, agents[1].sim_y, agents[1].sim_z, 1},
        {agents[2].sim_x, agents[2].sim_y, agents[2].sim_z, 2},
    };
    const int row_features = PARTNER_FEATURES + PARTNER_RELATIVE_VELOCITY_FEATURES;
    float obs[3 * (PARTNER_FEATURES + PARTNER_RELATIVE_VELOCITY_FEATURES)];
    int partner_count = 0;

    env.obs_slots_partners_n = 3;
    env.obs_partner_split_length_m = 0.0f;
    memset(obs, 0, sizeof(obs));
    int end_idx = write_partner_obs(&env, &agents[0], 0, partner_scan, obs, 0, &partner_count);
    EXPECT_EQ_INT(partner_count, 2);
    EXPECT_EQ_INT(end_idx, 3 * row_features);
    EXPECT_NEAR(obs[0], -0.1f, 1e-6f);
    EXPECT_NEAR(obs[row_features], 0.2f, 1e-6f);
    EXPECT_NEAR(obs[row_features + 1], 0.0f, 1e-6f);
    EXPECT_NEAR(obs[row_features + 3], 1.3f, 1e-6f);

    env.obs_partner_split_length_m = 7.0f;
    memset(obs, 0, sizeof(obs));
    end_idx = write_partner_obs(&env, &agents[0], 0, partner_scan, obs, 0, &partner_count);
    EXPECT_EQ_INT(partner_count, 3);
    EXPECT_EQ_INT(end_idx, 3 * row_features);
    EXPECT_NEAR(obs[0], -0.1f, 1e-6f);
    EXPECT_NEAR(obs[3], 0.4f, 1e-6f);
    EXPECT_NEAR(obs[row_features], 0.2f, 1e-6f);
    EXPECT_NEAR(obs[row_features + 1], -0.0325f, 1e-6f);
    EXPECT_NEAR(obs[row_features + 3], 0.65f, 1e-6f);
    EXPECT_NEAR(obs[2 * row_features], 0.2f, 1e-6f);
    EXPECT_NEAR(obs[2 * row_features + 1], 0.0325f, 1e-6f);
    EXPECT_EQ_INT(memcmp(&obs[row_features + 2], &obs[2 * row_features + 2], (row_features - 2) * sizeof(float)), 0);

    env.obs_slots_partners_n = 2;
    memset(obs, 0, sizeof(obs));
    end_idx = write_partner_obs(&env, &agents[0], 0, partner_scan, obs, 0, &partner_count);
    EXPECT_EQ_INT(partner_count, 2);
    EXPECT_EQ_INT(end_idx, 2 * row_features);
    EXPECT_NEAR(obs[0], -0.1f, 1e-6f);
    EXPECT_NEAR(obs[row_features + 1], -0.0325f, 1e-6f);
    return 0;
}

static int test_partner_obs_blindness_corridor_toggle(void) {
    // Ego heading +x, blind for the frame; car 20 m ahead in the corridor, car 20 m to the left outside it
    Drive env = {0};
    env.obs_norm_speed_mps = 10.0f;
    env.obs_norm_xy_offset_m = 100.0f;
    env.obs_norm_z_m = 10.0f;
    env.obs_norm_veh_length_m = 10.0f;
    env.obs_norm_veh_width_m = 5.0f;
    env.obs_range_partner_m = 100.0f;
    env.obs_slots_partners_n = 3;
    env.num_agents = 3;
    env.active_agent_count = 3;
    int active[3] = {0, 1, 2};
    env.active_agent_indices = active;
    Agent agents[3];
    agents[0] = drive_test_agent(0.0f, 0.0f, 0.0f);
    agents[1] = drive_test_agent(20.0f, 0.0f, 0.0f);
    agents[2] = drive_test_agent(0.0f, 20.0f, 0.0f);
    for (int agent_idx = 0; agent_idx < 3; agent_idx++) {
        update_agent_speed(&agents[agent_idx]);
    }
    env.agents = agents;
    PartnerScanEntry partner_scan[3] = {
        {agents[0].sim_x, agents[0].sim_y, agents[0].sim_z, 0},
        {agents[1].sim_x, agents[1].sim_y, agents[1].sim_z, 1},
        {agents[2].sim_x, agents[2].sim_y, agents[2].sim_z, 2},
    };
    float obs[3 * PARTNER_FEATURES];
    int partner_count = 0;

    env.partner_blindness_corridor = 1;
    agents[0].partner_blindness_counter = 2;
    memset(obs, 0, sizeof(obs));
    int end_idx = write_partner_obs(&env, &agents[0], 0, partner_scan, obs, 0, &partner_count);
    EXPECT_EQ_INT(partner_count, 1);
    EXPECT_EQ_INT(end_idx, 3 * PARTNER_FEATURES);
    EXPECT_EQ_INT(agents[0].partner_blindness_counter, 1);
    EXPECT_NEAR(obs[0], 0.2f, 1e-6f);
    EXPECT_NEAR(obs[1], 0.0f, 1e-6f);

    env.partner_blindness_corridor = 0;
    agents[0].partner_blindness_counter = 2;
    memset(obs, 0, sizeof(obs));
    end_idx = write_partner_obs(&env, &agents[0], 0, partner_scan, obs, 0, &partner_count);
    EXPECT_EQ_INT(partner_count, 0);
    EXPECT_EQ_INT(end_idx, 3 * PARTNER_FEATURES);
    EXPECT_EQ_INT(agents[0].partner_blindness_counter, 1);
    for (int i = 0; i < 3 * PARTNER_FEATURES; i++) {
        EXPECT_NEAR(obs[i], 0.0f, 0.0f);
    }

    agents[0].partner_blindness_counter = 0;
    memset(obs, 0, sizeof(obs));
    end_idx = write_partner_obs(&env, &agents[0], 0, partner_scan, obs, 0, &partner_count);
    EXPECT_EQ_INT(partner_count, 2);
    EXPECT_EQ_INT(end_idx, 3 * PARTNER_FEATURES);
    return 0;
}

static int test_partner_obs_blindness_facing_away(void) {
    // Blind ego heading +x; cars 20 m ahead driving away, oncoming, crossing, and one behind driving away
    Drive env = {0};
    env.obs_norm_speed_mps = 10.0f;
    env.obs_norm_xy_offset_m = 100.0f;
    env.obs_norm_z_m = 10.0f;
    env.obs_norm_veh_length_m = 10.0f;
    env.obs_norm_veh_width_m = 5.0f;
    env.obs_range_partner_m = 100.0f;
    env.obs_slots_partners_n = 4;
    env.num_agents = 5;
    env.active_agent_count = 5;
    env.partner_blindness_facing_away = 1;
    int active[5] = {0, 1, 2, 3, 4};
    env.active_agent_indices = active;
    Agent agents[5];
    agents[0] = drive_test_agent(0.0f, 0.0f, 0.0f);
    agents[1] = drive_test_agent(20.0f, 0.0f, 0.0f);
    agents[2] = drive_test_agent(20.0f, 3.5f, (float) M_PI);
    agents[3] = drive_test_agent(20.0f, -5.0f, 0.5f * (float) M_PI);
    agents[4] = drive_test_agent(-20.0f, 0.0f, (float) M_PI);
    PartnerScanEntry partner_scan[5];
    for (int agent_idx = 0; agent_idx < 5; agent_idx++) {
        update_agent_speed(&agents[agent_idx]);
        partner_scan[agent_idx]
            = (PartnerScanEntry) {agents[agent_idx].sim_x, agents[agent_idx].sim_y, agents[agent_idx].sim_z, agent_idx};
    }
    env.agents = agents;
    float obs[4 * PARTNER_FEATURES];
    int partner_count = 0;

    agents[0].partner_blindness_counter = 2;
    memset(obs, 0, sizeof(obs));
    write_partner_obs(&env, &agents[0], 0, partner_scan, obs, 0, &partner_count);
    EXPECT_EQ_INT(partner_count, 1);
    EXPECT_NEAR(obs[0], 0.2f, 1e-6f);
    EXPECT_NEAR(obs[1], 0.0f, 1e-6f);

    agents[0].partner_blindness_counter = 0;
    write_partner_obs(&env, &agents[0], 0, partner_scan, obs, 0, &partner_count);
    EXPECT_EQ_INT(partner_count, 4);
    return 0;
}

static int test_partner_blindness_episode_rate_floor(void) {
    Drive env = {0};
    env.init_seed = 7;
    env.partner_blindness_prob_min = 0.2f;
    env.partner_blindness_prob = 0.4f;
    for (int episode = 0; episode < 16; episode++) {
        begin_episode_rng(&env);
        EXPECT_TRUE(env.episode_partner_blindness_prob >= 0.2f);
        EXPECT_TRUE(env.episode_partner_blindness_prob <= 0.4f);
    }
    env.partner_blindness_prob_min = 0.4f;
    begin_episode_rng(&env);
    EXPECT_NEAR(env.episode_partner_blindness_prob, 0.4f, 1e-6f);
    return 0;
}

static int test_partner_blindness_time_fraction_scales_trigger_prob(void) {
    // 10-step bursts at trigger prob 0.03 leave a blind partner blind 10 / (10 + 0.97 / 0.03) = 23.6% of the time
    Drive env = {0};
    rng_seed(&env.rng_state, 7);
    env.episode_partner_blindness_prob = 1.0f;
    env.partner_blindness_trigger_prob = 0.03f;
    env.partner_blindness_duration = 10;
    Agent agent = drive_test_agent(0.0f, 0.0f, 0.0f);

    sample_erratic_flags(&env, &agent);
    EXPECT_EQ_INT(agent.is_blind_partner, 1);
    EXPECT_NEAR(agent.partner_blindness_trigger_prob, 0.03f, 1e-6f);

    float base_fraction = 10.0f * 0.03f / (10.0f * 0.03f + 1.0f - 0.03f);
    env.partner_blindness_time_fraction_max = base_fraction;
    sample_erratic_flags(&env, &agent);
    EXPECT_NEAR(agent.partner_blindness_trigger_prob, 0.03f, 1e-5f);

    env.partner_blindness_time_fraction_max = 1.0f;
    for (int draw = 0; draw < 32; draw++) {
        sample_erratic_flags(&env, &agent);
        EXPECT_EQ_INT(agent.is_blind_partner, 1);
        EXPECT_TRUE(agent.partner_blindness_trigger_prob >= 0.03f - 1e-6f);
        EXPECT_TRUE(agent.partner_blindness_trigger_prob <= 1.0f);
    }

    env.partner_blindness_trigger_prob = 1.0f;
    sample_erratic_flags(&env, &agent);
    EXPECT_NEAR(agent.partner_blindness_trigger_prob, 1.0f, 1e-6f);
    return 0;
}

static int test_partner_blindness_full_time_fraction_blinds_every_frame(void) {
    // Ego heading +x with trigger prob 1: the corridor hides the car on the left in every frame, the first included
    Drive env = {0};
    rng_seed(&env.rng_state, 7);
    env.obs_norm_speed_mps = 10.0f;
    env.obs_norm_xy_offset_m = 100.0f;
    env.obs_norm_z_m = 10.0f;
    env.obs_norm_veh_length_m = 10.0f;
    env.obs_norm_veh_width_m = 5.0f;
    env.obs_range_partner_m = 100.0f;
    env.obs_slots_partners_n = 3;
    env.num_agents = 3;
    env.active_agent_count = 3;
    env.partner_blindness_duration = 1;
    env.partner_blindness_corridor = 1;
    int active[3] = {0, 1, 2};
    env.active_agent_indices = active;
    Agent agents[3];
    agents[0] = drive_test_agent(0.0f, 0.0f, 0.0f);
    agents[1] = drive_test_agent(20.0f, 0.0f, 0.0f);
    agents[2] = drive_test_agent(0.0f, 20.0f, 0.0f);
    for (int agent_idx = 0; agent_idx < 3; agent_idx++) {
        update_agent_speed(&agents[agent_idx]);
    }
    env.agents = agents;
    agents[0].is_blind_partner = 1;
    agents[0].partner_blindness_trigger_prob = 1.0f;
    PartnerScanEntry partner_scan[3] = {
        {agents[0].sim_x, agents[0].sim_y, agents[0].sim_z, 0},
        {agents[1].sim_x, agents[1].sim_y, agents[1].sim_z, 1},
        {agents[2].sim_x, agents[2].sim_y, agents[2].sim_z, 2},
    };
    float obs[3 * PARTNER_FEATURES];
    int partner_count = 0;
    for (int frame = 0; frame < 20; frame++) {
        memset(obs, 0, sizeof(obs));
        write_partner_obs(&env, &agents[0], 0, partner_scan, obs, 0, &partner_count);
        EXPECT_EQ_INT(partner_count, 1);
    }

    agents[0].partner_blindness_trigger_prob = 0.0f;
    agents[0].partner_blindness_counter = 0;
    memset(obs, 0, sizeof(obs));
    write_partner_obs(&env, &agents[0], 0, partner_scan, obs, 0, &partner_count);
    EXPECT_EQ_INT(partner_count, 2);
    return 0;
}

static int test_reward_lane_align_wrong_way(void) {
    Drive env;
    Agent agent;
    Log log;
    int active[1];
    float reward[1] = {0};

    init_reward_env(&env, &agent, &log, active, reward);
    reward[0] = 0.0f;
    agent.metrics_array[LANE_ANGLE_IDX] = -1.0f; // cos(θ_f) = -1 → driving against lane
    agent.sim_speed_signed = 5.0f;
    agent.reward_coefs[REWARD_COEF_LANE_ALIGN] = 1.0f;
    agent.reward_coefs[REWARD_COEF_VEL_ALIGN] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_TRUE(log.reward_lane_align < 0.0f);
    return 0;
}

static int test_reward_reverse_fixed_penalty(void) {
    Drive env;
    Agent agent;
    Log log;
    int active[1];
    float reward[1] = {0};

    init_reward_env(&env, &agent, &log, active, reward);
    agent.sim_speed_signed = -1.0f;
    agent.reward_coefs[REWARD_COEF_REVERSE] = 0.01f;
    env.reward_reverse_fixed_per_second = 0.2f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(log.reward_reverse, -0.021f, 1e-6f);

    init_reward_env(&env, &agent, &log, active, reward);
    agent.sim_speed_signed = 1.0f;
    env.reward_reverse_fixed_per_second = 0.2f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(log.reward_reverse, 0.0f, 0.0f);
    return 0;
}

static RoadMapElement straight_test_lane(float *xs, float *ys, float *cum_lengths, float length_m) {
    RoadMapElement lane = {0};
    lane.segment_size = 2;
    lane.x = xs;
    lane.y = ys;
    lane.cum_lengths = cum_lengths;
    lane.length = length_m;
    return lane;
}

static int test_induced_braking_crossing_ego(void) {
    // Yielder routes east along y=0 at 10 m/s; ego routes north along x=15 and is 6 m short of the crossing at 5 m/s
    Drive env;
    Agent agents[2];
    Log logs[2];
    int active[2] = {0, 1};
    float reward[2] = {0};
    float path_xy[2 * ROUTE_PATH_MAX_POINTS * 2];
    int path_point_count[2];
    float junction_turn_rad[2];
    float junction_approach_heading[2];
    float east_x[2] = {-50.0f, 100.0f}, east_y[2] = {0.0f, 0.0f};
    float north_x[2] = {15.0f, 15.0f}, north_y[2] = {-50.0f, 100.0f};
    float cum_lengths[2] = {0.0f, 150.0f};
    RoadMapElement lanes[2] = {
        straight_test_lane(north_x, north_y, cum_lengths, 150.0f),
        straight_test_lane(east_x, east_y, cum_lengths, 150.0f),
    };
    int ego_route[1] = {0}, yielder_route[1] = {1};

    init_reward_env(&env, &agents[0], &logs[0], active, reward);
    memset(&logs[1], 0, sizeof(logs[1]));
    env.agents = agents;
    env.active_agent_count = 2;
    env.road_elements = lanes;
    env.route_path_xy = path_xy;
    env.route_path_point_count = path_point_count;
    env.route_junction_turn_rad = junction_turn_rad;
    env.route_junction_approach_heading = junction_approach_heading;
    env.reward_induced_braking = 0.5f;
    env.induced_braking_must_yield_scale = 3.0f;
    env.induced_braking_has_priority_scale = 0.0f;
    env.induced_braking_min_heading_cos = cosf(50.0f * (float) M_PI / 180.0f);
    env.induced_braking_threshold_mps2 = 1.0f;
    env.induced_braking_sample_count = 12;
    agents[0] = drive_test_agent(15.0f, -6.0f, (float) M_PI / 2.0f);
    agents[0].sim_speed = 5.0f;
    agents[0].route = ego_route;
    agents[0].route_length = 1;
    agents[1] = drive_test_agent(0.0f, 0.0f, 0.0f);
    agents[1].sim_speed = 10.0f;
    agents[1].route = yielder_route;
    agents[1].route_length = 1;
    predict_route_path(&env, 0);
    predict_route_path(&env, 1);
    predict_route_junction(&env, 0);
    predict_route_junction(&env, 1);
    EXPECT_EQ_INT(path_point_count[1], ROUTE_PATH_MAX_POINTS);
    EXPECT_EQ_INT(right_of_way(&env, 0, 1), RIGHT_OF_WAY_EQUAL);

    // worst sample is t = 1.75 s: 17.5 m of travel against 10 m of free road
    float must_yield_excess_mps2;
    EXPECT_NEAR(required_yield_decel_mps2(&env, 0, agents[0].sim_speed, 1, agents[1].sim_speed), 15.0f / 3.0625f, 1e-3f);
    EXPECT_NEAR(compute_induced_braking_excess_mps2(&env, 0, &must_yield_excess_mps2), 15.0f / 3.0625f - 1.0f, 1e-3f);
    EXPECT_NEAR(must_yield_excess_mps2, 0.0f, 0.0f);
    compute_rewards(&env, 0);
    EXPECT_NEAR(logs[0].reward_induced_braking, -0.5f * (15.0f / 3.0625f - 1.0f) * 0.1f, 1e-4f);
    EXPECT_NEAR(logs[0].induced_braking_seconds, 0.1f, 1e-6f);

    // a turning ego must yield to the straight car: scaled up; with the roles swapped ego has priority: free
    junction_turn_rad[0] = (float) M_PI / 2.0f;
    EXPECT_NEAR(
        compute_induced_braking_excess_mps2(&env, 0, &must_yield_excess_mps2), 3.0f * (15.0f / 3.0625f - 1.0f), 1e-3f);
    EXPECT_NEAR(must_yield_excess_mps2, 15.0f / 3.0625f - 1.0f, 1e-3f);
    junction_turn_rad[0] = 0.0f;
    junction_turn_rad[1] = -(float) M_PI / 2.0f;
    EXPECT_NEAR(compute_induced_braking_excess_mps2(&env, 0, &must_yield_excess_mps2), 0.0f, 0.0f);
    junction_turn_rad[1] = 0.0f;

    // yielder 3 m short of the crossing has its nose through it when ego arrives: ego cuts behind it
    agents[1].sim_x = 12.0f;
    predict_route_path(&env, 1);
    EXPECT_NEAR(required_yield_decel_mps2(&env, 0, agents[0].sim_speed, 1, agents[1].sim_speed), 0.0f, 0.0f);
    agents[1].sim_x = 0.0f;
    predict_route_path(&env, 1);

    // ego reaches the crossing after the horizon
    agents[0].sim_y = -30.0f;
    predict_route_path(&env, 0);
    EXPECT_NEAR(required_yield_decel_mps2(&env, 0, agents[0].sim_speed, 1, agents[1].sim_speed), 0.0f, 0.0f);

    // oncoming car in the opposite lane 3.5 m over never shares the path
    float west_x[2] = {100.0f, -50.0f}, west_y[2] = {3.5f, 3.5f};
    lanes[0] = straight_test_lane(west_x, west_y, cum_lengths, 150.0f);
    agents[0] = drive_test_agent(40.0f, 3.5f, (float) M_PI);
    agents[0].sim_speed = 10.0f;
    agents[0].route = ego_route;
    agents[0].route_length = 1;
    predict_route_path(&env, 0);
    EXPECT_NEAR(compute_induced_braking_excess_mps2(&env, 0, &must_yield_excess_mps2), 0.0f, 0.0f);
    return 0;
}

static int test_hesitation_only_when_clear(void) {
    // Ego drives east at 2 m/s on a straight 10 m/s lane; a second car and a red light take turns blocking it
    Drive env;
    Agent agents[2];
    Log logs[2];
    int active[2] = {0, 1};
    float reward[2] = {0};
    float path_xy[2 * ROUTE_PATH_MAX_POINTS * 2];
    int path_point_count[2];
    float lane_speed_limit_mps[2] = {10.0f, 10.0f};
    float east_x[2] = {-50.0f, 150.0f}, east_y[2] = {0.0f, 0.0f};
    float east_cum_lengths[2] = {0.0f, 200.0f};
    float north_x[2] = {500.0f, 500.0f}, north_y[2] = {-100.0f, 100.0f};
    RoadMapElement lanes[2] = {
        straight_test_lane(east_x, east_y, east_cum_lengths, 200.0f),
        straight_test_lane(north_x, north_y, east_cum_lengths, 200.0f),
    };
    int ego_route[1] = {0}, other_route[1] = {1};
    int light_states[1] = {TRAFFIC_CONTROL_STATE_RED};
    TrafficControlElement light = {0};
    light.type = TRAFFIC_CONTROL_TYPE_TRAFFIC_LIGHT;
    light.state_size = 1;
    light.states = light_states;
    float light_line[6] = {25.0f, -2.0f, 0.0f, 25.0f, 2.0f, 0.0f};
    memcpy(light.stop_line, light_line, sizeof(light_line));

    init_reward_env(&env, &agents[0], &logs[0], active, reward);
    memset(&logs[1], 0, sizeof(logs[1]));
    env.agents = agents;
    env.active_agent_count = 2;
    env.road_elements = lanes;
    env.lane_speed_limit_mps = lane_speed_limit_mps;
    env.route_path_xy = path_xy;
    env.route_path_point_count = path_point_count;
    env.base_max_speed_mps = 20.0f;
    env.reward_hesitation = 0.4f;
    env.induced_braking_min_heading_cos = cosf(50.0f * (float) M_PI / 180.0f);
    env.induced_braking_threshold_mps2 = 1.0f;
    env.induced_braking_sample_count = 12;
    env.traffic_lights_enabled = 1;
    agents[0] = drive_test_agent(0.0f, 0.0f, 0.0f);
    agents[0].sim_speed = 2.0f;
    agents[0].sim_speed_signed = 2.0f;
    agents[0].current_lane_idx = 0;
    agents[0].reward_coefs[REWARD_COEF_SPEED] = 1.0f;
    agents[0].route = ego_route;
    agents[0].route_length = 1;
    agents[1] = drive_test_agent(500.0f, 0.0f, (float) M_PI / 2.0f);
    agents[1].route = other_route;
    agents[1].route_length = 1;
    predict_route_path(&env, 0);
    predict_route_path(&env, 1);

    EXPECT_NEAR(compute_clear_target_speed_mps(&env, 0), 10.0f, 1e-4f);
    compute_rewards(&env, 0);
    EXPECT_NEAR(logs[0].reward_hesitation, -0.4f * (1.0f - 2.0f / 4.0f) * 0.1f, 1e-6f);
    EXPECT_NEAR(logs[0].hesitation_seconds, 0.1f, 1e-6f);

    // a stopped car 20 m ahead on ego's lane: ego at 10 m/s would have to brake for it
    agents[1] = drive_test_agent(20.0f, 0.0f, 0.0f);
    agents[1].route = ego_route;
    agents[1].route_length = 1;
    predict_route_path(&env, 1);
    EXPECT_EQ_INT(is_way_clear(&env, 0, 10.0f), 0);
    agents[1].sim_x = 200.0f;
    predict_route_path(&env, 1);
    EXPECT_EQ_INT(is_way_clear(&env, 0, 10.0f), 1);

    // red light 25 m ahead is within 3 s plus stopping distance at 10 m/s; green is not a block
    env.num_traffic_elements = 1;
    env.traffic_elements = &light;
    EXPECT_EQ_INT(is_way_clear(&env, 0, 10.0f), 0);
    light_states[0] = TRAFFIC_CONTROL_STATE_GREEN;
    EXPECT_EQ_INT(is_way_clear(&env, 0, 10.0f), 1);
    env.num_traffic_elements = 0;

    // a 12 m radius curve ahead caps the target at the comfortable cornering speed sqrt(3 * 12)
    float arc_x[41], arc_y[41], arc_cum_lengths[41];
    for (int point_idx = 0; point_idx <= 40; point_idx++) {
        float angle_rad = 0.5f * (float) M_PI * point_idx / 40.0f;
        arc_x[point_idx] = 12.0f * sinf(angle_rad);
        arc_y[point_idx] = 12.0f - 12.0f * cosf(angle_rad);
        arc_cum_lengths[point_idx] = 12.0f * angle_rad;
    }
    lanes[0].segment_size = 41;
    lanes[0].x = arc_x;
    lanes[0].y = arc_y;
    lanes[0].cum_lengths = arc_cum_lengths;
    lanes[0].length = 6.0f * (float) M_PI;
    predict_route_path(&env, 0);
    EXPECT_NEAR(compute_clear_target_speed_mps(&env, 0), 6.0f, 0.1f);
    return 0;
}

static int test_right_of_way_at_junction(void) {
    // Ego approaches east and turns left on a connector; the oncoming car approaches west 3.5 m to ego's left
    Drive env;
    Agent agents[2];
    Log logs[2];
    int active[2] = {0, 1};
    float reward[2] = {0};
    float junction_turn_rad[2];
    float junction_approach_heading[2];
    float approach_x[2] = {-50.0f, 0.0f}, approach_y[2] = {0.0f, 0.0f};
    float connector_x[3] = {0.0f, 10.0f, 10.0f}, connector_y[3] = {0.0f, 0.0f, 10.0f};
    float connector_headings[3] = {0.0f, (float) M_PI / 2.0f, (float) M_PI / 2.0f};
    float approach_cum_lengths[2] = {0.0f, 50.0f}, connector_cum_lengths[3] = {0.0f, 10.0f, 20.0f};
    int approach_exits[2] = {1, 1};
    int connector_entries[1] = {0};
    RoadMapElement lanes[2] = {
        straight_test_lane(approach_x, approach_y, approach_cum_lengths, 50.0f),
        straight_test_lane(connector_x, connector_y, connector_cum_lengths, 20.0f),
    };
    lanes[0].num_exits = 2;
    lanes[0].exit_lanes = approach_exits;
    lanes[1].segment_size = 3;
    lanes[1].headings = connector_headings;
    lanes[1].num_entries = 1;
    lanes[1].entry_lanes = connector_entries;
    int ego_route[2] = {0, 1};

    init_reward_env(&env, &agents[0], &logs[0], active, reward);
    memset(&logs[1], 0, sizeof(logs[1]));
    env.agents = agents;
    env.active_agent_count = 2;
    env.road_elements = lanes;
    env.route_junction_turn_rad = junction_turn_rad;
    env.route_junction_approach_heading = junction_approach_heading;
    env.induced_braking_min_heading_cos = cosf(50.0f * (float) M_PI / 180.0f);
    env.reward_collision_must_yield = 1.5f;
    agents[0] = drive_test_agent(-5.0f, 0.0f, 0.0f);
    agents[0].route = ego_route;
    agents[0].route_length = 2;
    agents[1] = drive_test_agent(30.0f, 3.5f, (float) M_PI);
    predict_route_junction(&env, 0);
    predict_route_junction(&env, 1);
    EXPECT_NEAR(junction_turn_rad[0], (float) M_PI / 2.0f, 1e-5f);
    EXPECT_NEAR(junction_approach_heading[0], 0.0f, 0.0f);
    EXPECT_NEAR(junction_turn_rad[1], 0.0f, 0.0f);

    EXPECT_EQ_INT(right_of_way(&env, 0, 1), RIGHT_OF_WAY_EGO_YIELDS);
    EXPECT_EQ_INT(right_of_way(&env, 1, 0), RIGHT_OF_WAY_EGO_PRIORITY);

    // oncoming right-turner merges into ego's exit: the left-turner turns across it and still yields
    junction_turn_rad[1] = -(float) M_PI / 2.0f;
    EXPECT_EQ_INT(right_of_way(&env, 0, 1), RIGHT_OF_WAY_EGO_YIELDS);
    EXPECT_EQ_INT(right_of_way(&env, 1, 0), RIGHT_OF_WAY_EGO_PRIORITY);

    // two oncoming left-turners, or two straight cars, have no rule between them
    junction_turn_rad[1] = (float) M_PI / 2.0f;
    EXPECT_EQ_INT(right_of_way(&env, 0, 1), RIGHT_OF_WAY_EQUAL);
    junction_turn_rad[0] = 0.0f;
    junction_turn_rad[1] = 0.0f;
    EXPECT_EQ_INT(right_of_way(&env, 0, 1), RIGHT_OF_WAY_EQUAL);

    // moving left-turner hits the oncoming car mid-turn: extra penalty; the oncoming car pays none
    predict_route_junction(&env, 0);
    agents[0].sim_heading = (float) M_PI / 4.0f;
    agents[0].cos_heading = cosf(agents[0].sim_heading);
    agents[0].sin_heading = sinf(agents[0].sim_heading);
    agents[0].sim_speed = 4.0f;
    agents[0].first_collision_partner_idx = 1;
    agents[0].metrics_array[COLLISION_IDX] = 1.0f;
    compute_rewards(&env, 0);
    EXPECT_NEAR(logs[0].reward_collision_must_yield, -1.5f, 1e-6f);
    EXPECT_NEAR(logs[0].must_yield_collision_rate, 1.0f, 0.0f);
    agents[1].sim_speed = 4.0f;
    agents[1].first_collision_partner_idx = 0;
    EXPECT_EQ_INT(is_must_yield_collision(&env, 1), 0);
    agents[0].sim_speed = 0.0f;
    EXPECT_EQ_INT(is_must_yield_collision(&env, 0), 0);
    return 0;
}

static int test_off_route_fires_once_per_departure(void) {
    // Route = one straight 100 m lane along +x; off route beyond 5 m for 3 steps
    Drive env;
    Agent agent;
    Log log;
    int active[1];
    float reward[1] = {0};
    float lane_x[2] = {0.0f, 100.0f};
    float lane_y[2] = {0.0f, 0.0f};
    float lane_cum_lengths[2] = {0.0f, 100.0f};
    RoadMapElement lane = {0};
    lane.segment_size = 2;
    lane.x = lane_x;
    lane.y = lane_y;
    lane.cum_lengths = lane_cum_lengths;
    lane.length = 100.0f;
    int route[1] = {0};

    init_reward_env(&env, &agent, &log, active, reward);
    env.road_elements = &lane;
    env.goal_source = GOAL_SOURCE_ROUTE;
    env.reward_off_route = 1.0f;
    env.off_route_distance_m = 5.0f;
    env.off_route_steps = 3;
    agent.route = route;
    agent.route_length = 1;
    agent.sim_x = 50.0f;

    agent.sim_y = 3.0f;
    update_off_route(&env, &agent);
    EXPECT_EQ_INT(agent.off_route_step_count, 0);
    agent.sim_y = 8.0f;
    for (int step = 0; step < 5; step++) {
        agent.off_route_triggered = 0;
        update_off_route(&env, &agent);
        EXPECT_EQ_INT(agent.off_route_triggered, step == 2);
        if (agent.off_route_triggered) {
            compute_rewards(&env, 0);
        }
    }
    EXPECT_NEAR(log.reward_off_route, -1.0f, 1e-6f);
    EXPECT_NEAR(log.off_route_count, 1.0f, 0.0f);

    agent.sim_y = 1.0f;
    update_off_route(&env, &agent);
    EXPECT_EQ_INT(agent.off_route_step_count, 0);
    agent.sim_y = -9.0f;
    for (int step = 0; step < 3; step++) {
        update_off_route(&env, &agent);
    }
    EXPECT_EQ_INT(agent.off_route_triggered, 1);
    return 0;
}

static int test_lane_goal_distance_obs_unreachable_reads_far(void) {
    const float norm_m = LANE_GRAPH_DISTANCE_NORM_M;
    float dest[2];

    write_lane_goal_distance_obs(0.25f * norm_m, 0.5f * norm_m, dest);
    EXPECT_NEAR(dest[0], 0.25f, 1e-6f);
    EXPECT_NEAR(dest[1], -0.25f, 1e-6f);

    // Unreachable lane next to an ego lane 3 norms from the goal: must read worse than ego, not closer.
    write_lane_goal_distance_obs(INFINITY, 3.0f * norm_m, dest);
    EXPECT_NEAR(dest[0], 1.0f, 1e-6f);
    EXPECT_NEAR(dest[1], 1.0f, 1e-6f);

    write_lane_goal_distance_obs(3.0f * norm_m, INFINITY, dest);
    EXPECT_NEAR(dest[0], 1.0f, 1e-6f);
    EXPECT_NEAR(dest[1], -1.0f, 1e-6f);

    write_lane_goal_distance_obs(INFINITY, INFINITY, dest);
    EXPECT_NEAR(dest[0], 1.0f, 1e-6f);
    EXPECT_NEAR(dest[1], 0.0f, 1e-6f);

    write_lane_goal_distance_obs(0.25f * norm_m, -1.0f, dest);
    EXPECT_NEAR(dest[0], 0.25f, 1e-6f);
    EXPECT_NEAR(dest[1], 0.0f, 1e-6f);

    write_lane_goal_distance_obs(-1.0f, 0.5f * norm_m, dest);
    EXPECT_NEAR(dest[0], 0.0f, 1e-6f);
    EXPECT_NEAR(dest[1], 0.0f, 1e-6f);
    return 0;
}

static int test_lane_graph_distance_to_goal_sentinels(void) {
    Drive env = drive_test_make_env(drive_carla_map(), SIMULATION_MODE_GIGAFLOW, 1, 0);
    EXPECT_TRUE(env.lane_graph.n_lanes >= 2);
    int lane_idx = env.lane_graph.lane_ids[0];
    int goal_graph_idx = 1;
    float *stored_distance_m = &env.lane_graph.distances[goal_graph_idx * env.lane_graph.n_lanes + 0];

    *stored_distance_m = 123.0f;
    EXPECT_NEAR(lane_graph_distance_to_goal_m(&env, lane_idx, goal_graph_idx), 123.0f, 1e-6f);
    *stored_distance_m = INFINITY;
    EXPECT_TRUE(isinf(lane_graph_distance_to_goal_m(&env, lane_idx, goal_graph_idx)));
    *stored_distance_m = -1.0f;
    EXPECT_TRUE(isinf(lane_graph_distance_to_goal_m(&env, lane_idx, goal_graph_idx)));
    EXPECT_TRUE(lane_graph_distance_to_goal_m(&env, -1, goal_graph_idx) < 0.0f);
    EXPECT_TRUE(lane_graph_distance_to_goal_m(&env, env.num_road_elements, goal_graph_idx) < 0.0f);

    free_allocated(&env);
    return 0;
}

int main(void) {
    int failures = 0;
    RUN_TEST(test_lane_goal_distance_obs_unreachable_reads_far);
    RUN_TEST(test_lane_graph_distance_to_goal_sentinels);
    RUN_TEST(test_observation_size_formula);
    RUN_TEST(test_lane_speed_limit_column);
    RUN_TEST(test_observation_zero_fill_and_valid_counts);
    RUN_TEST(test_reward_terminal_components);
    RUN_TEST(test_reward_goal_speed_gating);
    RUN_TEST(test_reward_coef_goal_speed_pinned);
    RUN_TEST(test_partner_obs_relative_velocity);
    RUN_TEST(test_partner_obs_split_long_partner);
    RUN_TEST(test_partner_obs_blindness_corridor_toggle);
    RUN_TEST(test_partner_obs_blindness_facing_away);
    RUN_TEST(test_partner_blindness_episode_rate_floor);
    RUN_TEST(test_partner_blindness_time_fraction_scales_trigger_prob);
    RUN_TEST(test_partner_blindness_full_time_fraction_blinds_every_frame);
    RUN_TEST(test_reward_lane_align_wrong_way);
    RUN_TEST(test_reward_reverse_fixed_penalty);
    RUN_TEST(test_off_route_fires_once_per_departure);
    RUN_TEST(test_induced_braking_crossing_ego);
    RUN_TEST(test_right_of_way_at_junction);
    RUN_TEST(test_hesitation_only_when_clear);
    return test_summary(failures);
}
