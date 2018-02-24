import os
import sys
import numpy as np
import random
import scipy.io.wavfile

# fpath = '/home/maneesh/Downloads/IEMOCAP_full_release/Session'
#ex-/home/maneesh/Downloads/IEMOCAP_full_release/Session3/sentences/wav/
#ex-/home/maneesh/Downloads/IEMOCAP_full_release/Session3/dialog/transcriptions
#ex-/home/maneesh/Downloads/IEMOCAP_full_release/Session3/dialog/wav

def extract_data(fpath):
    emotion_label = {}
    map_emlab_num = {'ang':0,'dis':1,'exc':2, 'fea':3, 'fru':4, 'hap':5,'neu':6,'sad':7, 'sur':8, 'xxx':9, 'oth':9}
    sp_count, tr_count = 0, 0
    MIN = 10000000
    MAX = 0
    len_list = []
    [resize_type, width] = ['mean',5000]# append_max, truncate_min, truncate_mean
    for sess in range(1,6):
        for folders in enumerate(os.listdir(fpath+str(sess)+'/sentences/wav/')):
            for aud_files in os.listdir(os.path.join(fpath+str(sess)+'/sentences/wav/', folders[1])):
                em_file = open(os.path.join(fpath+str(sess),'dialog','EmoEvaluation', folders[1]+'.txt'),'r')
                with open(os.path.join(fpath + str(sess),'sentences', 'wav', folders[1], aud_files),'r') as f:
                    lines = em_file.readlines()
                    for line in lines:# read only the emotion label for file & avoid rubbish
                        if line[0] == '[':
                            dialog_name, em_label = line.split('\t')[1], line.split('\t')[2]
                            emotion_label[dialog_name] = map_emlab_num.get(em_label)
                    # print np.unique(emotion_label.values())
                    try:
                        _rate, temp = scipy.io.wavfile.read(f)
                        len_list.append(len(temp))
                        if len(temp) < MIN:
                            MIN = len(temp)
                        if len(temp) > MAX:
                            MAX = len(temp)
                        sp_count += 1
                    except:
                        print sys.exc_info()[0]
    MEAN = int(sum(len_list)/len(len_list))
    print MIN, MAX, len(emotion_label.keys())
    sp_count = 0
    dialogs = {}
    for sess in range(1,6):
        for folders in enumerate(os.listdir(fpath+str(sess)+'/sentences/wav/')):
            for aud_files in os.listdir(os.path.join(fpath+str(sess)+'/sentences/wav/', folders[1])):
                with open(os.path.join(fpath + str(sess),'sentences', 'wav', folders[1], aud_files),'r') as f:
                    try:
                        _rate, temp = scipy.io.wavfile.read(f)
                    except:
                        print sys.exc_info()[0]
                    if resize_type == 'append_max': # append to the MAX size in the dataset
                            temp = np.append(temp, np.zeros((MAX-temp.shape[0],1)))
                            temp = np.reshape(temp,(1,temp.shape[0]))
                    elif resize_type == 'truncate_min': # truncate to the MIN size
                        temp = temp[0:MIN]
                        temp = np.reshape(temp, (1, temp.shape[0]))
                    elif resize_type == 'truncate_mean': # get width no. of samples about MEAN
                        temp = np.reshape(temp, (1, temp.shape[0]))
                        mid_len = int(np.floor(temp.shape[1]/2))
                        temp = temp[0,mid_len-width:mid_len+width]
                    elif resize_type == 'mean': # adjust all size to have MEAN size
                        temp = np.reshape(temp, (1, temp.shape[0]))
                        if temp.shape[1] < MEAN:
                            temp = np.append(temp, np.zeros((1,MEAN-temp.shape[1])))
                        else:
                            temp = temp[0,0:MEAN]
                    dialogs[aud_files.split('.')[0]] = temp
                    sp_count += 1

    print len(dialogs.keys()), sp_count, MEAN, temp.shape
    # check any possible mismatches between emotion_label and dialogs
    missing_flag, miss_count = 0, 0

    for key in dialogs.keys():
        if emotion_label.get(key) == None:
            missing_flag = 1
            miss_count += 1
            # print key

    speech_data = np.array(list(dialogs.values()))
    label = np.array(list(emotion_label.values()))
    print 'presence of missing data:{0}'.format(missing_flag)
    if missing_flag:
        print 'number of missing samples:{0}'.format(missing_count)
    print speech_data.shape, label.shape#len(label)
    np.savez('iemocap_emotion', data = speech_data, label = label, ref_label = map_emlab_num)

# only select some classes
    labels_list = [0, 1, 2, 3, 4, 5]
    print data.shape
    if not all_classes:  # 3 move this to extract function in utils
        for unlabel in [1, 3, 6, 9]:
            ind = np.where(targets == unlabel)
            targets = np.delete(targets, ind)
            data = np.delete(data, ind, axis=0)
        print data.shape, targets.shape
        for lab in enumerate(np.unique(targets)):
            ind = np.where(targets == lab[1])
            targets[ind[0]] = labels_list[lab[0]]
        print np.unique(targets)

    return 0

def batch(data_len, bsize):
    indices = range(data_len)
    random.shuffle(indices)
    sindex = 0
    eindex = bsize-1
    while eindex < len(indices):
        batch = indices[sindex:eindex]
        sindex = eindex
        eindex = eindex + bsize
        yield batch

    if eindex > data_len:
        batch = indices[sindex:]
        yield batch

# for sess in range(1,6):
    # for transcript in enumerate(os.listdir(os.path.join(fpath+str(sess), 'dialog/transcriptions/'))):
    #     for line in open(fpath+str(sess)+'/dialog/transcriptions/'+transcript[1],'r'):
    #         try:
    #             splt_name = line.split('[')[1].split(']')
    #             # txt = np.append(txt, splt_name[0:10])
    #             start_time,end_time = splt_name[0].split('-')
    #             tr_count += 1
    #             # print start_time,end_time,len(x),float(start_time)*16000,float(end_time)*16000
    #             # scipy.io.wavfile.write('./sample/temp', _rate, x[int(float(start_time))*16000:int(float(end_time)*16000)])
    #         except:
    #             print sys.exc_info()[0]


# for sess in range(1,6):
#     for transcript in enumerate(os.listdir(fpath+str(sess)+'/dialog/transcriptions/')):
#         sp_file = fpath+str(sess)+'/dialog/wav/'+transcript[1].split('.')[0]+'.wav'
#         count += 1
#         try:
#             _rate, x = scipy.io.wavfile.read(sp_file)
#         except Exception as e:
#             print 'Unexpected error:',sys.exc_info()[0],e
#         for line in open(fpath+str(sess)+'/dialog/transcriptions/'+transcript[1],'r'):
#             try:
#                 splt_name = line.split('[')[1].split(']')
#                 start_time,end_time = splt_name[0].split('-')
#                 # print start_time,end_time,len(x),float(start_time)*16000,float(end_time)*16000
#                 scipy.io.wavfile.write('./sample/temp', _rate, x[int(float(start_time))*16000:int(float(end_time)*16000)])
#             except:
#                 print sys.exc_info()[0]
#         # scipy.io.wavfile.write('temp', _rate, x)
#         # print _rate,len(x)#,_rate#.shape
# print count