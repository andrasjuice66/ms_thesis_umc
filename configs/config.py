import yaml
import os
from pathlib import Path


class Config:
    """
    This class loads configuration from a YAML file and provides
    methods to access configuration values.
    """
    
    def __init__(self, config_path):
        """Initialize configuration from YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config(config_path)
    
    def _load_config(self, config_path):
        """Load configuration from YAML file.
        
        Args:
            config_path (str): Path to the YAML configuration file
            
        Returns:
            dict: Configuration dictionary
        """
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    
    def get(self, key, default=None):
        """Get configuration value by key.
        
        Supports nested keys with dot notation (e.g., 'model.type').
        
        Args:
            key (str): Configuration key
            default: Default value to return if key is not found
            
        Returns:
            Configuration value or default if key is not found
        """
        if '.' not in key:
            return self.config.get(key, default)
        
        # Handle nested keys
        parts = key.split('.')
        value = self.config
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value
    
    def save_config(self, output_path):
        """Save configuration to a YAML file.
        
        Args:
            output_path (str or Path): Path to save the configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def __str__(self):
        """String representation of the configuration.
        
        Returns:
            str: String representation
        """
        return str(self.config)
    
    def __repr__(self):
        """Representation of the configuration.
        
        Returns:
            str: Representation
        """
        return f"Config(path={self.config_path})"
