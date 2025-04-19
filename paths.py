# defines file paths for the given task

def get_paths(task):
	paths = {
		'treecover': {
			'home_dir': '/hopkilau/CNNS_for_SI/',
			'img_dir': '/datadrive/mosaiks/data/png/contus_uar_128/',
			'log_dir': '/datadrive/mosaiks/treecover/runs/cutmix/',
			'model_dir': '/datadrive/mosaiks/treecover/models/cutmix/',
			'means': '/hopkilau/CNNS_for_SI/channel_means/channel_means_contus_uar_Apr-06-2022.txt',
		},

		'nightlights': {
			'home_dir': '/hopkilau/CNNS_for_SI/',
			'img_dir': '/datadrive/mosaiks/data/png/contus_pop_128/',
			'log_dir': '/datadrive/mosaiks/nightlights/runs/cutmix/',
			'model_dir': '/datadrive/mosaiks/nightlights/models/cutmix/',
			'means': '/hopkilau/CNNS_for_SI/channel_means/channel_means_contus_pop_Nov-16-2022.txt',
		},

		'elevation': {
			'home_dir': '/hopkilau/CNNS_for_SI/',
			'img_dir': '/datadrive/mosaiks/data/png/contus_uar_128/',
			'log_dir': '/datadrive/mosaiks/elevation/runs/cutmix/',
			'model_dir': '/datadrive/mosaiks/elevation/models/cutmix/',
			'means': '/hopkilau/CNNS_for_SI/channel_means/channel_means_contus_uar_Apr-06-2022.txt',
		},

		'landuse': {
			'home_dir': '/hopkilau/CNNS_for_SI/',
			'img_dir': '/datadrive/ucMerced_landuse/data/npy/',
			'log_dir': '/datadrive/ucMerced_landuse/runs/cutmix/',
			'model_dir': '/datadrive/ucMerced_landuse/models/cutmix/',
			'means': '/hopkilau/CNNS_for_SI/channel_means/channel_means_ucMerced_landuse_Jan-05-2023.txt',
		},

		'coffee': {
			'home_dir': '/hopkilau/CNNS_for_SI/',
			'img_dir': '/datadrive/coffee/data/jpg/',
			'log_dir': '/datadrive/coffee/runs/cutmix/',
			'model_dir': '/datadrive/coffee/models/cutmix/',
			'means': '/hopkilau/CNNS_for_SI/channel_means/channel_means_coffee_Aug-18-2024.txt',
		},

		'eurosat': {
			'home_dir': '/hopkilau/CNNS_for_SI/',
			'img_dir': '/datadrive/eurosat/data/eurosat_ms/npy/',
			'log_dir': '/datadrive/eurosat/runs/cutmix/',
			'model_dir': '/datadrive/eurosat/models/cutmix/',
			'means': '/hopkilau/CNNS_for_SI/channel_means/channel_means_eurosat_ms_Jul-01-2024.txt',
		}
	}

	return paths[task]

