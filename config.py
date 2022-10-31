class Config:
    n_agents = 2

    comm_fail_prob = 0
    comm_fail_period = 5

    scheme = "maac_4agents_800epi"
    csv_filename_prefix = None
    model_filename_prefix = None
    scheme_template = 'hac_{}agents_{}comm'
    # scheme_template = 'hac_{}agents_{}comm_{}period'
    experiment_prefix = './results/'
    csv_filename_prefix = '/save/statistics'
    csv_filename_prefix_template = '/save/statistics-{}'
    model_filename_prefix_template = '/model/model-{}'
    discrete = False

    random_seed = 2021
    memory_size = 100000
    batch_size = 32
    grid_width = 20
    grid_height = 20
    fov = 45
    xyreso = 1
    yawreso = 5
    sensing_range = 8
    # n_obstacles = 10
    n_targets = 10

    episodes = 800
    max_step = 50
    checkpoint_interval = 50
    update_freq = 250
    gamma = 0.99
    lr_actor = 0.001
    lr_critic = 0.005

    num_options = 5

    hidden_size = 512

    @classmethod
    def update(cls):
        cls.scheme = cls.scheme_template.format(cls.n_agents, cls.comm_fail_prob, cls.comm_fail_period)
        cls.csv_filename_prefix = cls.csv_filename_prefix_template.format(cls.scheme)
        cls.model_filename_prefix = cls.model_filename_prefix_template.format(cls.scheme)