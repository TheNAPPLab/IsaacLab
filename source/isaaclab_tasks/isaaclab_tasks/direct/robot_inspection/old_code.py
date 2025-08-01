        # if debug:
        #     print(f"✅ Tracking {len(self.target_prim_map)} target objects: {self.target_object_names}")
        # self._setup_semantics()

 # def _setup_semantics(self):
    #     """Setup semantic tags using the semantic manager."""
    #     try:
    #         print("Setting up semantic tags from configuration...")
    #         success = add_semantic_tags_from_config(self.cfg.semantic_config_path)
    #         if success:
    #             if debug:
    #                 print("✅ Successfully applied semantic tags from configuration")
    #             # Initialize semantic manager for runtime use
    #             self.semantic_manager = SemanticManager(self.cfg.semantic_config_path)
    #         else:
    #             if debug:
    #                 print("❌ Failed to apply some or all semantic tags")

    #     except Exception as e:
    #         if debug:
    #             print(f"❌ ERROR: Failed to setup semantics: {e}")
    #             print("Continuing without semantic tags...")
    #         pass



        # self.action_scale = self.cfg.action_scale
        # self.semantic_manager = SemanticManager(self.cfg.semantic_config_path, initialize_stage=False)
        # self.target_prim_map = {
        #     obj["prim_path"]: obj["object_name"]
        #     for obj in self.semantic_manager.config.get("semantic_objects", [])
        # }
        # self.target_object_names = list(self.target_prim_map.values())



   def get_comprehensive_mesh_metadata(self, mesh_prim_path: str) -> dict:
        """Extract complete mesh metadata from current scene for reward design."""
        from pxr import UsdGeom
        import omni.usd
        
        # Get the current stage from the running simulation
        stage = omni.usd.get_context().get_stage()
        mesh_prim = stage.GetPrimAtPath(mesh_prim_path)
        
        if not mesh_prim.IsValid():
            print(f"Warning: Mesh prim not found at {mesh_prim_path}")
            return {"total_faces": 0, "total_vertices": 0}
            
        mesh = UsdGeom.Mesh(mesh_prim)
        face_count = mesh.GetFaceCount()
        face_counts = mesh.GetFaceVertexCountsAttr().Get()
        face_indices = mesh.GetFaceVertexIndicesAttr().Get()
        points = mesh.GetPointsAttr().Get()
        
        return {
            "total_faces": len(face_counts) if face_counts else 0,
            "total_vertices": len(points) if points else 0,
            "face_vertex_counts": face_counts,
            "face_vertex_indices": face_indices,
            # "bounding_box": UsdGeom.Boundable(mesh_prim).ComputeWorldBound(0, "default")
        }