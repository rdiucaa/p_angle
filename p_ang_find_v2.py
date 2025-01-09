
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:15:52 2024

@author: rushikesh
"""


import math
import datetime
import spiceypy 
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import os



# bc_file_name = '21_dec/SUT81N18P1AL10013308NNNN23355050409296_UNP_9999_999999_00_qib.bc'
# bc_file_name = '17_may_2024/SUT81N18P1AL10022808NNNN24139075037459_T24_0728_000381_00_qib.bc'
# bc_file_name = '27_may_2024/SUT81N18P1AL10021908NNNN24149071702658_T24_0774_000391_00_qib.bc'
# bc_file_name = '18_june_2024_full/al1_spice_att_18june.bc'

# bc_file_name = '21_june_2024/SUT81N18P1AL10005708NNNN24174072212502_T24_0876_000423_00_qib.bc'
# bc_file_name = '20_sept_2024/SUT81N18P1AL10000108NNNN24265073807855_T24_1310_000543_00_qib.bc'
# bc_file_name = '31_july_2024/SUT81D32P1AL10000608NNNN24214081056023_T24_1022_000490_00_qib.bc'
# bc_file_name = '22_aug_2024/SUT81D32P1AL10003408NNNN24236081135165_C24_0329_000500_00_qib.bc'
bc_file_name = '30_oct_2024/SUT81N18P1AL10019108NNNN24305053620769_C24_0409_000630_00_qib.bc'
# bc_file_name = '01_oct_2024/SUT81N18P1AL10000008NNNN24276063310634_T24_1457_000577_00_qib.bc'
# bc_file_name = '12_sept_2024/SUT81N18P1AL10003808NNNN24257072939579_C24_0338_000509_00_qib.bc'

# bc_file_name = 'al1_june_full_spice_attitude.bc'



# bsp_file_name = '21_dec/SUTXXN18P1AL10013308NNNN23355050409296_UNP_9999_999999_V1_1/traj_master_al1.bsp'
# bsp_file_name = '21_dec/SUTXXN18P1AL10013308NNNN23355050409296_UNP_9999_999999_V1_1/traj_master_al1.bsp'
# bsp_file_name = '17_may_2024/SUTXXN18P1AL10022808NNNN24139075037459_T24_0728_000381_V1_1/traj_master_al1.bsp'
# bsp_file_name = '27_may_2024/SUTXXN18P1AL10021908NNNN24149071702658_T24_0774_000391_V1_1/traj_master_al1.bsp'
# bsp_file_name = '18_june_2024_full/full_day_bsp/traj_master_al1.bsp'
# bsp_file_name = '21_june_2024/SUTXXN18P1AL10005708NNNN24174072212502_T24_0876_000423_V1_1/traj_master_al1.bsp'
# bsp_file_name = '20_sept_2024/SUTXXN18P1AL10000108NNNN24265073807855_T24_1310_000543_V1_1/traj_master_al1.bsp'
# bsp_file_name = '31_july_2024/SUTXXD32P1AL10000608NNNN24214081056023_T24_1022_000490_V1_1/traj_master_al1.bsp'
# bsp_file_name = '22_aug_2024/SUTXXD32P1AL10003408NNNN24236081135165_C24_0329_000500_V1_1/traj_master_al1.bsp'
# bsp_file_name = '30_oct_2024/SUTXXN18P1AL10019108NNNN24305053620769_C24_0409_000630_V1_1/traj_master_al1.bsp'
# bsp_file_name = '01_oct_2024/SUTXXN18P1AL10000008NNNN24276063310634_T24_1457_000577_V1_1/traj_master_al1.bsp'
# bsp_file_name = '12_sept_2024/SUTXXN18P1AL10003808NNNN24257072939579_C24_0338_000509_V1_1/traj_master_al1.bsp'
# bsp_file_name = '01/

# utc_fits = '2024-05-17T22:30:00.000'
# utc_fits = '2024-05-27T23:03:00.000'
# utc_fits = '2023-12-20T21:30:00.000'
# utc_fits = '2024-06-18T03:56:02.493'
# utc_fits = '2024-06-21T11:56:02.493'
# utc_fits = '2024-09-20T21:42:49.656'
# utc_fits = '2024-07-31T00:02:49.656'
# utc_fits = '2024-08-22T00:20:49.656'
utc_fits = '2024-10-30T01:20:49.656'
# utc_fits = '2024-10-01T06:20:49.656'
# utc_fits = '2024-09-12T00:42:49.656'


def find_p_ang(bc,utc_fits):
    
    sc_id = -156001;    #spacecraft NAIF ID
    sc_obj_id = -156;
    
    spk = '../kernels/spk/de432s.bsp'
    pck = '../kernels/pck/pck00011_n0066.tpc'
    # sclk = '../kernels/sclk/solo_ANC_soc-sclk_20231003_V01.tsc'
    sclk = '../kernels/sclk/clockgen_sclk_al1.tsc'
    bc = '../kernels/ck/'+ bc_file_name 
    # bc = '/home/rushikesh/Downloads/SUIT_files/SPICE/p_angle_problem/nov_some_days_bc/' + bc_file_name
    # bsp = '../kernels/spk/'+ bsp_file_name
    lsk = '../kernels/lsk/naif0012.tls.txt'   
    fk = '../kernels/fk/ADITYA_frame_kernel_v03.tf' 
    # fk = '../kernels/fk/ADITYA_frame_kernel_v04.tf' 
    
            
    if spk:
        spiceypy.furnsh(spk);
    else:
        print('SPK not found!');
    
    if pck:
        spiceypy.furnsh(pck);
    else:
        print('PCK not found!');    
    if sclk:
        spiceypy.furnsh(sclk);
    else:
        print('SCLK not found');
    if bc:
        spiceypy.furnsh(bc);
    else:
        print('BC not found');
    # if bsp:
    #     spiceypy.furnsh(bsp);
    # else:
    #     print('BSP not found!');
    if lsk:
        spiceypy.furnsh(lsk);
    else:
        print('LSK not found!');
    if fk:
        spiceypy.furnsh(fk);
    else:
        print('LSK not found!');    
        
        
    if spk and pck and sclk and bc and lsk and fk :
    # if spk and pck and sclk and bc and bsp and lsk and fk :
        print('all kernels present')
        
        # string UTC to et conversion
        et_now = spiceypy.str2et(utc_fits)
        
        int_et_now = int(et_now)
        print('int_et_now = ',int_et_now)
        
        # C file coverage
        coverage = spiceypy.ckcov ( bc,sc_id,False,"SEGMENT",  0.0,  "TDB")
        # print('coverage =',coverage)
        
        # coverage of C kernel
        begin_c,end_c = spiceypy.wnfetd(coverage,0)
        # print('begin_c = ', begin_c)
        begin_c_int = int(begin_c)
        # print('end_c = ', end_c)
    
        cov_c = int(end_c-begin_c)
        print('c = ',cov_c)
        
        t_vec = np.linspace(coverage[0],coverage[1],cov_c);
        print('len_tvec = ',len(t_vec))
        t_req = int_et_now - begin_c_int

        
        utc_start,utc_end = spiceypy.et2utc(coverage, 'ISOC', 3);
        print('utc_start = ', utc_start)
        print('utc_start = ', utc_end)
        
        print('t_req = ',t_req)
        print(t_vec[t_req])
        
        # divide the et duration into small intervals and find the state and other values:
        t_bin = 1; #in seconds
        t_vec = np.linspace(coverage[0],coverage[1],int((coverage[1]- coverage[0])/t_bin));
    
        
        if  begin_c < et_now < end_c :
            print('within ck coverage')
            
           
           
            #define suit axes in its frame:
            vec_yaw_suit = np.array([1.0, 0.0, 0.0]);   vec_roll_suit = np.array([0.0, 1.0, 0.0]);vec_pitch_suit = np.array([0.0, 0.0, 1.0])
            
            # solar north in HCI frame....Z axis
            hci_sun_north = np.array([0.0, 0.0, 1.0]);
            # yaw2_angle_deg = np.zeros([len(t_vec),1]); roll2_angle_deg = np.zeros([len(t_vec),1]); pitch2_angle_deg = np.zeros([len(t_vec),1]);
            
            utc_time=spiceypy.et2utc(t_vec, 'ISOC', 3);
            datetime_obj=[datetime.strptime(time,'%Y-%m-%dT%H:%M:%S.%f') for time in utc_time]
            # print('datetime_obj = ',datetime_obj)

            # start_time = min(datetime_obj)
            # end_time = start_time + timedelta(minutes=5)
            
    
                
            # cmat2 = spiceypy.pxform('ADITYA_SUIT2','HCI',t_vec[t_req]);
           
    
            # vec_yaw_sun = spiceypy.mxv(cmat2,vec_yaw_suit );
            # yaw_angle_radian = spiceypy.vsep(vec_yaw_sun, hci_sun_north);
            # yaw2_angle_deg = spiceypy.dpr()*yaw_angle_radian
            # print('yaw_angle =',  yaw2_angle_deg)
            
            # vec_roll_sun = spiceypy.mxv(cmat2,vec_roll_suit );
            # roll_angle_radian = spiceypy.vsep(vec_roll_sun, hci_sun_north);
            # roll2_angle_deg = spiceypy.dpr()*roll_angle_radian
            # # print('p_angle =',  roll2_angle_deg)
            
            
            # vec_pitch_sun = spiceypy.mxv(cmat2,vec_pitch_suit );
            # pitch_angle_radian = spiceypy.vsep(vec_pitch_sun, hci_sun_north);
            # pitch2_angle_deg = spiceypy.dpr()*pitch_angle_radian;
            # print('pitch2_angle_deg =',  pitch2_angle_deg)
           
            
           # modified method for p-angle:
           # Calculate everything in SUIT CCD frame
            cmat3 = spiceypy.pxform('HCI','ADITYA_SUIT2',t_vec[t_req]);
            # cmat3 = spiceypy.pxform('HCI','ADITYA',t_vec[t_req]);
            # get solar north in SUIT frame using the above transformation
            solar_north_in_suitframe = spiceypy.mxv(cmat3,hci_sun_north );
            print('solar_north_in_suitframe = ', solar_north_in_suitframe)
            # project the solar north in suitframe onto the CCD plane :
            # considering that the CCD plane is closely aligned with ROLL-PITCH plane and YAW is the boresight
            solar_north_suitccd = np.array([0,solar_north_in_suitframe[1],solar_north_in_suitframe[2]]);
            print('solar_north_suitccd = ', solar_north_suitccd)

                
            # the separation between suit roll vector(which should align with CCD columns) and solar_north_suitccd : p-angle
            p_angle_radian = spiceypy.vsep(vec_roll_suit, solar_north_suitccd);
            print('p_angle_radian = ', p_angle_radian)
            p_angle_deg = spiceypy.dpr()*p_angle_radian
            # print('p_angle =',  p_angle_deg)
            print(solar_north_suitccd[2])
            if float(solar_north_suitccd[2]) < 0:
                p_angle_deg = -1 * p_angle_deg
            print('p_angle =',  p_angle_deg)
            
            spiceypy.kclear()
            return(p_angle_deg)    
         
# p_ang =  find_p_ang(bc_file_name,bsp_file_name,utc_fits)       
p_ang =  find_p_ang(bc_file_name,utc_fits)          
            
            