# occiput 
# Stefano Pedemonte
# Aalto University, School of Science, Helsinki
# Oct 2013, Helsinki 
# Martinos Center for Biomedical Imaging, Harvard University/MGH, Boston
# Dec. 2013, Boston
# Martinos Center for Biomedical Imaging, Harvard University/MGH, Boston
# Jan. 2015, Boston
# June 2015, Helsinki 
# Nov. 2015 - Mar. 2017, Boston 

from time import time


class ReconstructionProfiler():
    """Keeps a record of the computing time for the reconstruction, projection and backprojection."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.T_rec_reset()
        self.T_project_reset()
        self.T_backpro_reset()
        self.total = 0.0

    def projection(self):
        return self.T_project

    def backprojection(self):
        return self.T_backpro

    def reconstruction(self):
        self.T_rec['T00_total'] = self.total
        return self.T_rec

    def tic(self):
        self._time = time()

    def tac(self):
        elapsed = (time() - self._time) * 1000.0
        self.total += elapsed
        return elapsed

    def rec_iteration(self):
        self.T_rec["N_iteations"] += 1

    def rec_uncompress(self):
        T = self.tac()
        self.T_rec['T13_uncompress'] += T

    def rec_get_subset_attenuation(self):
        T = self.tac()
        self.T_rec['T18_get_subset_attenuation'] += T

    def rec_get_subset_sensitivity(self):
        T = self.tac()
        self.T_rec['T17_get_subset_sensitivity'] += T

    def rec_get_subset_randoms(self):
        T = self.tac()
        self.T_rec['T15_get_subset_randoms'] += T

    def rec_get_subset_prompts(self):
        T = self.tac()
        self.T_rec['T16_get_subset_prompts'] += T

    def rec_get_subset_scatter(self):
        T = self.tac()
        self.T_rec['T14_get_subset_scatter'] += T

    def rec_compose_various(self):
        T = self.tac()
        self.T_rec['T21_compose_various'] += T

    def rec_compose_randoms(self):
        T = self.tac()
        self.T_rec['T19_compose_randoms'] += T

    def rec_compose_scatter(self):
        T = self.tac()
        self.T_rec['T20_compose_scatter'] += T

    def rec_backprojection_norm_total(self):
        T = self.tac()
        self.T_rec['T22_backprojection_norm_total'] += T

    def rec_backprojection_activity_total(self):
        T = self.tac()
        self.T_rec['T12_backprojection_activity_total'] += T

    def rec_projection_activity_total(self):
        T = self.tac()
        self.T_rec['T11_projection_activity_total'] += T

    def rec_update(self):
        T = self.tac()
        self.T_rec['T23_update'] += T

    def rec_project_projection(self):
        T = self.tac()
        self.T_project['t1_project_total'] = T

    def rec_project_make_continuous(self):
        T = self.tac()
        self.T_project['t2_make_continuous'] = T

    def rec_project_get_subset_sparsity(self):
        T = self.tac()
        self.T_project['t3_get_subset_sparsity'] = T

    def rec_project_scale(self):
        T = self.tac()
        self.T_project['t4_scale'] = T

    def rec_project_wrap(self):
        T = self.tac()
        self.T_project['t5_wrap'] = T

    def rec_project_get_angles(self):
        T = self.tac()
        self.T_project['t6_get_angles'] = T

    def rec_project_exponentiate(self):
        T = self.tac()
        self.T_project['t7_exponentiate'] = T

    def rec_backpro_backprojection(self):
        T = self.tac()
        self.T_backpro['t1_backproject_total'] = T

    def rec_backpro_get_subset(self):
        T = self.tac()
        self.T_backpro['t2_get_subset'] = T

    def rec_backpro_get_subset_data(self):
        T = self.tac()
        self.T_backpro['t3_get_data_subset'] = T

    def rec_backpro_get_subset_sparsity(self):
        T = self.tac()
        self.T_backpro['t4_get_subset_sparsity'] = T

    def rec_backpro_scale(self):
        T = self.tac()
        self.T_backpro['t5_scale'] = T

    def rec_backpro_wrap(self):
        T = self.tac()
        self.T_backpro['t6_wrap'] = T

    def rec_backpro_get_angles(self):
        T = self.tac()
        self.T_backpro['t7_get_angles'] = T

    def T_project_reset(self):
        self.T_project = {
            'T0_transfer_to_gpu': 0.0,
            'T1_alloc': 0.0,
            'T2_rotate': 0.0,
            'T3_resample': 0.0,
            'T4_integral': 0.0,
            'T5_transfer_to_host': 0.0,
            'T6_free': 0.0,
            't1_project_total': 0.0,
            't2_make_continuous': 0.0,
            't3_get_subset_sparsity': 0.0,
            't4_scale': 0.0,
            't5_wrap': 0.0,
            't6_get_angles': 0.0,
            't7_exponentiate': 0.0}

    def T_backpro_reset(self):
        self.T_backpro = {
            'T0_transfer_to_gpu': 0.0,
            'T1_alloc': 0.0,
            'T2_rotate': 0.0,
            'T3_resample': 0.0,
            'T4_integral': 0.0,
            'T5_transfer_to_host': 0.0,
            'T6_free': 0.0,
            'T7_accumulate': 0.0,
            'T8_clear_memory': 0.0,
            'T9_copy_texture': 0.0,
            't1_backproject_total': 0.0,
            't2_get_subset': 0.0,
            't3_get_data_subset': 0.0,
            't4_get_subset_sparsity': 0.0,
            't5_scale': 0.0,
            't6_wrap': 0.0,
            't7_get_angles': 0.0}

    def T_rec_reset(self):
        self.T_rec = {}
        self.T_rec["N_iteations"] = 0
        self.T_rec["N_projections"] = 0
        self.T_rec["N_backprojections"] = 0
        self.T_rec["T00_total"] = 0
        self.T_rec["T01_project_transfer"] = 0.0
        self.T_rec["T02_backpro_transfer"] = 0.0
        self.T_rec["T03_project_alloc"] = 0.0
        self.T_rec["T04_backpro_alloc"] = 0.0
        self.T_rec["T05_project_integral"] = 0.0
        self.T_rec["T06_backpro_integral"] = 0.0
        self.T_rec["T07_project_rotate"] = 0.0
        self.T_rec["T08_backpro_rotate"] = 0.0
        self.T_rec["T09_project_resample"] = 0.0
        self.T_rec["T10_backpro_resample"] = 0.0
        self.T_rec["T11_projection_activity_total"] = 0.0
        self.T_rec["T12_backprojection_activity_total"] = 0.0
        self.T_rec["T13_uncompress"] = 0.0
        self.T_rec["T14_get_subset_scatter"] = 0.0
        self.T_rec["T15_get_subset_randoms"] = 0.0
        self.T_rec["T16_get_subset_prompts"] = 0.0
        self.T_rec["T17_get_subset_sensitivity"] = 0.0
        self.T_rec["T18_get_subset_attenuation"] = 0.0
        self.T_rec["T19_compose_randoms"] = 0.0
        self.T_rec["T20_compose_scatter"] = 0.0
        self.T_rec["T21_compose_various"] = 0.0
        self.T_rec["T22_backprojection_norm_total"] = 0.0
        self.T_rec["T23_update"] = 0.0

    def rec_projection(self, p):
        self.T_rec['N_projections'] += 1
        self.T_project['T0_transfer_to_gpu'] = p['T0_transfer_to_gpu']
        self.T_project['T1_alloc'] = p['T1_alloc']
        self.T_project['T2_rotate'] = p['T2_rotate']
        self.T_project['T3_resample'] = p['T3_resample']
        self.T_project['T4_integral'] = p['T4_integral']
        self.T_project['T5_transfer_to_host'] = p['T5_transfer_to_host']
        self.T_project['T6_free'] = p['T6_free']

        self.T_rec["T01_project_transfer"] += p['T0_transfer_to_gpu'] + p['T5_transfer_to_host']
        self.T_rec["T03_project_alloc"] += p['T1_alloc'] + p['T6_free']
        self.T_rec["T05_project_integral"] += p['T4_integral']
        self.T_rec["T07_project_rotate"] += p['T2_rotate']
        self.T_rec["T09_project_resample"] += p['T3_resample']

    def rec_backprojection(self, b):
        self.T_rec['N_backprojections'] += 1
        self.T_backpro['T0_transfer_to_gpu'] = b['T0_transfer_to_gpu']
        self.T_backpro['T1_alloc'] = b['T1_alloc']
        self.T_backpro['T2_rotate'] = b['T2_rotate']
        self.T_backpro['T3_resample'] = b['T3_resample']
        self.T_backpro['T4_integral'] = b['T4_integral']
        self.T_backpro['T5_transfer_to_host'] = b['T5_transfer_to_host']
        self.T_backpro['T6_free'] = b['T6_free']
        self.T_backpro['T7_accumulate'] = b['T7_accumulate']
        self.T_backpro['T8_clear_memory'] = b['T8_clear_memory']
        self.T_backpro['T9_copy_texture'] = b['T9_copy_texture']

        self.T_rec["T02_backpro_transfer"] += b['T0_transfer_to_gpu'] + b['T9_copy_texture'] + b['T5_transfer_to_host']
        self.T_rec["T04_backpro_alloc"] += b['T1_alloc'] + b['T6_free'] + b['T8_clear_memory']
        self.T_rec["T06_backpro_integral"] += b['T4_integral']
        self.T_rec["T08_backpro_rotate"] += b['T2_rotate']
        self.T_rec["T10_backpro_resample"] += b['T3_resample']
