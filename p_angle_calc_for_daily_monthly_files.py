#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 14:20:30 2025

@author: rushikesh

# new p angle calculator script

DESCRIPTION
- Find P-angle of image based on UTC of image.
- Inputs- * UTC time of image.
          * Camera Kernel from specific data packet. This function can process both daily and monthly .bc files
- Output- (float) CROTA2 in degrees. -ve value means image is rotated clockwise.

If the queried time is not in the coverage the code looks for nearest coverage time and provides the p angle of that time.
Checked correct output for monthly and daily files 
"""

import numpy as np
import spiceypy


utc_fits = '2025-05-3T00:22:16.873'


# File paths

bc_file_name = '/home/rushikesh/Downloads/SUIT_files/SPICE/kernels/ck/03_may_2025/SUT81N18P1AL10013008NNNN25124042236913_C25_0113_000910_00_qib.bc'

def calc_p_ang_daily_monthly(bc_file_name,utc_fits):
    
    try:
        # Clear any previous SPICE kernels
        spiceypy.kclear()
        # Load necessary kernels
        spiceypy.furnsh(bc_file_name)
        spiceypy.furnsh('../kernels/spk/de432s.bsp')
        spiceypy.furnsh('../kernels/pck/pck00011_n0066.tpc')
        spiceypy.furnsh('../kernels/sclk/clockgen_sclk_al1.tsc')
        spiceypy.furnsh('../kernels/lsk/naif0012.tls.txt')
        spiceypy.furnsh('../kernels/fk/ADITYA_frame_kernel_v05.tf')
        
        # Get coverage time
        coverage = spiceypy.ckcov(bc_file_name, -156001, False, "SEGMENT", 0.0, "TDB")
        card_num = coverage.card
        # print('card_num =', card_num)

        def create_cov_arrays(card_num):
            # Create the start_cov_arr with numbers from 0 to card_num - 2, with a step of 2
            start_cov_arr = list(range(0, card_num, 2))
            # Create the end_cov_arr with numbers from 1 to card_num - 1, with a step of 2
            end_cov_arr = list(range(1, card_num, 2))
            return start_cov_arr, end_cov_arr
        
        cov_start,cov_end = create_cov_arrays(card_num) # coverage index list
        # print('cov_start = ', cov_start)
        # print('cov_end = ', cov_end)
        utc_start_t_list = []
        cov_end_t_list = []
        
        for i,j in zip(cov_start,cov_end): # list of et corresponding to the start and end times of a coverage window
            utc_start_t_list.append(coverage[i])
            cov_end_t_list.append(coverage[j])
            
        # print('utc_start_t_list = ', utc_start_t_list)
        # print('cov_end_t_list = ', cov_end_t_list)
        
        
        
        et_now = spiceypy.str2et(utc_fits) # UTC to et conversion
        # print('et_now = ', et_now)
        
        def find_or_closest(x, list1, list2):
            # First pass: check if x lies strictly between each pair
            for a, b in zip(list1, list2):
                if a < x < b:
                    return x
        
            # Second pass: check between list2[n] and list1[n+1]
            best = 'not_in_coverage_range'
            best_dist = float('inf')
            for i in range(len(list2) - 1):
                lo = list2[i]
                hi = list1[i + 1]
                if lo < x < hi:
                    # determine which bound is closest to x
                    d_lo = x - lo
                    d_hi = hi - x
                    candidate = lo if d_lo <= d_hi else hi
                    dist = d_lo if d_lo <= d_hi else d_hi
                    if dist < best_dist:
                        best_dist = dist
                        best = candidate
        
            return best
        
        # find the closest time to the input utc time if the input time does 
        # not fall within the coverage (for monthly files with more than 1 coverage windows)
        res = find_or_closest(et_now, utc_start_t_list, cov_end_t_list)  
        if res != 'not_in_coverage_range':
            corrected_et = res
        else:
            print(res)
            
        
        vec_roll_suit = np.array([0.0, 1.0, 0.0])
        hci_sun_north = np.array([0.0, 0.0, 1.0])  # Solar north in HCI frame (Z-axis)
        
        
        # utc = spiceypy.et2utc(corrected_et, 'ISOC',5)
        # print('utc = ', utc)
        
        # Transform HCI to SUIT frame
        cmat3 = spiceypy.pxform('HCI', 'ADITYA', corrected_et)
        solar_north_in_suitframe = spiceypy.mxv(cmat3, hci_sun_north)
        
        # Project onto CCD plane
        solar_north_suitccd = np.array([0, solar_north_in_suitframe[1], solar_north_in_suitframe[2]])
        
        # Calculate P angle
        p_angle_radian = spiceypy.vsep(vec_roll_suit, solar_north_suitccd)
        p_angle_deg = spiceypy.dpr() * p_angle_radian
        if solar_north_suitccd[2] < 0: # positive and neg p angle correction as per the orbit
            p_angle_deg = -p_angle_deg
        
        
        print('p_angle_deg = ', p_angle_deg)
        spiceypy.kclear()
        return p_angle_deg
    except:
        return 'incorrect_bc_kernel_file'
    
if __name__ == "__main__":   
    p_angle_deg = calc_p_ang_daily_monthly(bc_file_name,utc_fits)    

    
