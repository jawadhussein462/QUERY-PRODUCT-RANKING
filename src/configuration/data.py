"""Retrieve all variables from files configurations'."""

import json
import logging
import os


class DataConfig:
    """Retrieve information from the configuration files.
    There are located in "global_scoring_model/configuration/resources"
    Methods:
    --------
    _prepare_data_structure_configuration
    _data_structure_filepath
    _model_filepath
    _cdp_filepath
    _path_filepath
    """

    _input_path = os.path.join(os.path.dirname(__file__), "resources")
    _data_structure_filepath = os.path.join(_input_path, "data_structure.json")
    _model_filepath = os.path.join(_input_path, "model.json")
    _path_filepath = os.path.join(_input_path, "path.json")

    def __init__(self):
        """Initialize data configuration."""
        self._prepare_data_structure_configuration()
        self._prepare_model_configuration()
        self._prepare_path_configuration()

    def _prepare_data_structure_configuration(self):
        """Load the data structure configuration."""
        logging.info("  - Loading 'data structure' configuration attributes")
        with open(self._data_structure_filepath) as json_file:
            self.data_structure = json.load(json_file)

    def _prepare_model_configuration(self):
        """Load the scoring model configuration."""
        logging.info("  - Loading 'scoring model' configuration attributes")
        with open(self._model_filepath) as json_file:
            self.model = json.load(json_file)

    def _prepare_path_configuration(self):
        """Load the path configuration."""
        logging.info("  - Loading 'path' configuration attributes")
        with open(self._path_filepath) as json_file:
            self.path = json.load(json_file)
