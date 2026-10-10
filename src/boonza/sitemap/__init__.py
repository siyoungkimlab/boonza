"""SiteMap-style binding sites of coarse-grained proteins (Martini 2, Martini 3, SIRAH).

SiteMap (Halgren 2007, 2009) finds sites as grid points outside a protein that
are enclosed by it and in good contact with a probe, grouped and merged, and
ranks them by SiteScore.  Here the same on beads: the protein mapped by boonza
(or a run's own beads), a grid whose per-point numbers are computed once
(``grid``), site finding as thresholds and grouping on them (``sites``), with
each model's parameters and SiteScore weights tuned on a static training set
(``presets``, data/sitemap/presets.json).  A trajectory's sites are grouped
into pockets by the residues lining them (``traj``), shown with the probes'
pharmacophore hotspots (``pharm``) as a table and a PyMOL view (``report``);
one structure's the same (``structure``).

    boonza sitemap traj --workdir run/md_solute --model martini3 -o sitemap/
    boonza sitemap structure apo.pdb --model martini3 -o sitemap/ --holo holo.pdb
"""
