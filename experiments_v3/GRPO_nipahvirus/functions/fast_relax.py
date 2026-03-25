relax_me(
    pdb_in=list(pdb_dict.keys())[0],
    pdb_out="relaxed.pdb",
    max_iterations=max_iterations,
    tolerance=tolerance,
    stiffness=stiffness,
    use_gpu=use_gpu
)