#!/usr/bin/env python3

import json
import os
from typing import Dict, List, Any
import isaacsim.core.utils.stage as stage_utils

try:
    import Semantics
except ModuleNotFoundError:
    from pxr import Semantics


class SemanticManager:
    """Manages semantic tags for objects in Isaac Sim based on JSON configuration."""
    
    def __init__(self, config_path: str = "semantic_config.json", initialize_stage: bool = True):
        """
        Initialize the semantic manager.
        
        Args:
            config_path: Path to the JSON configuration file
            initialize_stage: Whether to initialize the stage connection (set False for config-only use)
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.stage = stage_utils.get_current_stage() if initialize_stage else None
        self.applied_objects = []
    
    def _load_config(self) -> Dict[str, Any]:
        """Load semantic configuration from JSON file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Semantic config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = json.load(f)
        
        # print(f"Loaded semantic config with {len(config['semantic_objects'])} objects")
        return config
    
    @staticmethod
    def get_semantic_filter_from_config(config_path: str) -> str:
        """
        Get semantic filters from config file without initializing stage.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            Comma-separated string of semantic filters
        """
        try:
            if not os.path.exists(config_path):
                print(f"Warning: Semantic config file not found: {config_path}")
                return "class:traffic_cone"  # Default fallback
            
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            filters = config.get("camera_filters", ["class:traffic_cone"])
            return "; ".join(filters) if len(filters) > 1 else (filters[0] if filters else "class:traffic_cone")
            
        except Exception as e:
            # print(f"Warning: Could not load semantic config, using default filter: {e}")
            return "class:traffic_cone"
    
    def add_semantic_tags_to_prim(self, prim_path: str, semantic_tags: List[Dict[str, str]], 
                                  object_name: str = None) -> bool:
        """
        Add semantic tags to a specific prim.
        
        Args:
            prim_path: USD prim path
            semantic_tags: List of semantic tag dictionaries with 'type' and 'value'
            object_name: Optional object name for logging
            
        Returns:
            True if successful, False otherwise
        """
        if self.stage is None:
            print("ERROR: Stage not initialized. Create SemanticManager with initialize_stage=True")
            return False
            
        prim = self.stage.GetPrimAtPath(prim_path)
        
        if not prim.IsValid():
            print(f"WARNING: Prim at {prim_path} not found")
            return False
        
        success_count = 0
        for tag in semantic_tags:
            try:
                semantic_type = tag["type"]
                semantic_value = tag["value"]
                instance_name = f"{semantic_type}_{semantic_value}"
                
                # Apply semantic API
                sem = Semantics.SemanticsAPI.Apply(prim, instance_name)
                sem.CreateSemanticTypeAttr().Set(semantic_type)
                sem.CreateSemanticDataAttr().Set(semantic_value)
                success_count += 1
                
            except Exception as e:
                print(f"ERROR: Failed to add semantic tag {tag} to {prim_path}: {e}")
        
        object_display_name = object_name or prim_path
        print(f"Applied {success_count}/{len(semantic_tags)} semantic tags to {object_display_name}")
        return success_count > 0
    
    def apply_all_semantics(self) -> Dict[str, bool]:
        """
        Apply semantic tags to all objects defined in the configuration.
        
        Returns:
            Dictionary mapping object names to success status
        """
        results = {}
        
        for obj_config in self.config["semantic_objects"]:
            prim_path = obj_config["prim_path"]
            object_name = obj_config["object_name"]
            semantic_tags = obj_config["semantic_tags"]
            
            success = self.add_semantic_tags_to_prim(
                prim_path=prim_path,
                semantic_tags=semantic_tags,
                object_name=object_name
            )
            
            results[object_name] = success
            if success:
                self.applied_objects.append(object_name)
        
        return results
    
    def get_camera_filters(self) -> List[str]:
        """Get camera semantic filters from configuration."""
        return self.config.get("camera_filters", [])
    
    def get_semantic_filter_string(self) -> str:
        """Get semantic filters as a single string for camera configuration."""
        filters = self.get_camera_filters()
        return ",".join(filters) if len(filters) > 1 else (filters[0] if filters else "")
    
    def check_prim_exists(self, prim_path: str) -> bool:
        """Check if a prim exists at the given path."""
        if self.stage is None:
            print("WARNING: Stage not initialized, cannot check prim existence")
            return False
        prim = self.stage.GetPrimAtPath(prim_path)
        return prim.IsValid()
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate the current configuration by checking if prims exist.
        
        Returns:
            Validation results
        """
        validation_results = {
            "valid_prims": [],
            "invalid_prims": [],
            "total_objects": len(self.config["semantic_objects"])
        }
        
        for obj_config in self.config["semantic_objects"]:
            prim_path = obj_config["prim_path"]
            object_name = obj_config["object_name"]
            
            if self.check_prim_exists(prim_path):
                validation_results["valid_prims"].append({
                    "name": object_name,
                    "path": prim_path
                })
            else:
                validation_results["invalid_prims"].append({
                    "name": object_name,
                    "path": prim_path
                })
        
        return validation_results
    
    def print_validation_report(self):
        """Print a validation report of the configuration."""
        results = self.validate_config()
        
        print("\n" + "="*60)
        print("SEMANTIC CONFIGURATION VALIDATION REPORT")
        print("="*60)
        print(f"Total objects in config: {results['total_objects']}")
        print(f"Valid prims found: {len(results['valid_prims'])}")
        print(f"Invalid prims: {len(results['invalid_prims'])}")
        
        if results['valid_prims']:
            print("\n✅ VALID PRIMS:")
            for obj in results['valid_prims']:
                print(f"  - {obj['name']}: {obj['path']}")
        
        if results['invalid_prims']:
            print("\n❌ INVALID PRIMS:")
            for obj in results['invalid_prims']:
                print(f"  - {obj['name']}: {obj['path']}")
        
        print("="*60 + "\n")


# Convenience function for backward compatibility
def add_semantic_tags_from_config(config_path: str = "semantic_config.json") -> bool:
    """
    Load semantic configuration and apply all tags.
    
    Args:
        config_path: Path to the JSON configuration file
        
    Returns:
        True if any semantics were successfully applied
    """
    try:
        manager = SemanticManager(config_path)
        manager.print_validation_report()
        results = manager.apply_all_semantics()
        
        success_count = sum(1 for success in results.values() if success)
        total_count = len(results)
        
        print(f"\nSemantic application summary: {success_count}/{total_count} objects successful")
        return success_count > 0
        
    except Exception as e:
        print(f"ERROR: Failed to apply semantics from config: {e}")
        return False