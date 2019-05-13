import unittest

import blind_detection

class TestDetectionDecision(unittest.TestCase):

    def test_dm_time_box(self):
        """ make sure that dm_time_box method passes 
        basic logic test
        """
        DD = blind_detection.DetectionDecision(100., 1924.)
        decision1 = DD.dm_time_box_decision(101., 1924.01)
        decision2 = DD.dm_time_box_decision(101., 1924.1, t_err=0)#1e-3)
        decision3 = DD.dm_time_box_decision(101., 1924.01, dm_err=1e-3, t_err=1e-3)

        self.assertTrue(decision1)
        self.assertFalse(decision2)
        self.assertFalse(decision3)

    def test_three_decision_methods(self):
        """ make sure each method of a decision 
        (bowtie contour, gaussian contour, dm/time box) all give 
        expected answers
        """
        DD = blind_detection.DetectionDecision(1000., 42.)

        decision1 = DD.dm_time_contour_decision(1000., 42., dmtarr_function='bowtie', simulator='injectfrb')[0]
        decision2 = DD.dm_time_contour_decision(1000., 42., dmtarr_function='box', simulator='injectfrb')[0]
        decision3 = DD.dm_time_contour_decision(1000., 42., dmtarr_function='gaussian', simulator='injectfrb')[0]

        self.assertTrue(decision1)
        self.assertTrue(decision2)
        self.assertTrue(decision3)

        decision1 = DD.dm_time_contour_decision(1000., 33., dmtarr_function='bowtie', simulator='injectfrb')[0]
        decision2 = DD.dm_time_contour_decision(1000., 50., dmtarr_function='box', simulator='injectfrb')[0]
        decision3 = DD.dm_time_contour_decision(1000., 15., dmtarr_function='gaussian', simulator='injectfrb')[0]

        self.assertFalse(decision1)
        self.assertFalse(decision2)
        self.assertFalse(decision3)

    def test_get_decision_array(self):
        """ test that the decision array method runs 
        for different contours (box and gaussian)
        """
        fn_truth = './examples/truth.txt'
        fn_amber = './examples/amber_example.trigger'

        blind_detection.get_decision_array(fn_truth, fn_amber, dmtarr_function='box', 
                                            freq_ref_truth=1400., freq_ref_cand=1400., mk_plot=False)
        blind_detection.get_decision_array(fn_truth, fn_amber, dmtarr_function='gaussian', 
                                            freq_ref_truth=1400., freq_ref_cand=1400., mk_plot=False)


if __name__ == '__main__':
    unittest.main()












    