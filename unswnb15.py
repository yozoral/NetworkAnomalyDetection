import logging
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


def get_shape_input():
    """Get shape of the dataset for UNSW-NB15"""
    return (None, 41)


def get_shape_label():
    """Get shape of the labels in UNSW-NB15"""
    return (None,)


def get_anomalous_proportion():
    return 13


def get_test():
    if not os.path.exists("data/unswnb15/UNSW-NB15_test.csv"):
        return None

    df_test = pd.read_csv("data/unswnb15/UNSW-NB15_test.csv")
    return to_xy(df_test, target='label')


def get_valid():
    if os.path.exists("data/unswnb15/UNSW-NB15_valid.csv"):
        return "data/unswnb15/UNSW-NB15_valid.csv"
    return None


def get_train():
    if os.path.exists("data/unswnb15/UNSW-NB15_training.csv"):
        return "data/unswnb15/UNSW-NB15_training.csv"
    return None

def _prepare_dataset():
    """ Gets the basic dataset
    Returns :
            dataset (dict): containing the data
                dataset['x_train'] (np.array): training images shape
                (?, 206)
                dataset['y_train'] (np.array): training labels shape
                (?,)
                dataset['x_test'] (np.array): testing images shape
                (?, 206)
                dataset['y_test'] (np.array): testing labels shape
                (?,)
    """
    
    #all_files = ["data/unswnb15/UNSW_NB15_training-set.csv", "data/unswnb15/UNSW_NB15_testing-set.csv"]
    all_files = ["./unsw/UNSW-NB15_1.csv","./unsw/UNSW-NB15_2.csv","./unsw/UNSW-NB15_3.csv","./unsw/UNSW-NB15_4.csv"]
    col_names = _col_names()
    df = pd.concat((pd.read_csv(f, header=None, names=col_names) for f in all_files), ignore_index=True)

    df.drop(['srcip', 'sport', 'dstip', 'dsport','Stime','Ltime','attack_cat'], axis=1, inplace=True)

    df.replace({np.nan:0}, inplace=True)
    df.replace({' ':0}, inplace=True)

    df['ct_ftp_cmd'] = df['ct_ftp_cmd'].astype(int)
    #num_keys = df.iloc[:, :-2].select_dtypes(exclude=['object']).keys()
    #cat_keys = df.iloc[:, :-2].select_dtypes(include=['object']).keys() # this includes 'ct_fpt_cmd'
    num_keys = ['dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'Sload',
           'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
           'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit', 'Sintpkt',
           'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
           'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_srv_src',
           'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm', 'ct_src_dport_ltm', 'ct_ftp_cmd']

    cat_keys = ['proto', 'state', 'service']

    #log transform for dataset
    #df[num_keys] = df[num_keys].apply(lambda x: np.log(x+1))

    # Minmax transform for dataset
    scaler = MinMaxScaler()
    df[num_keys] = scaler.fit_transform(df[num_keys])

    for key in cat_keys:
        # encoding using label encoding
        df[key] = pd.Series(data=pd.factorize(df[key])[0])
        # encoding using frequency
        #fe = df.groupby(key).size()/len(df)
        #df.loc[:, key+'_freq_encode'] = df[key].map(fe)
        #df.drop(key, axis=1, inplace=True)

    #scaler = MinMaxScaler()
    #df[cat_keys] = scaler.fit_transform(df[cat_keys])

    #print("Number of data points: " + str(len(df.index)))
    #print("Number of anormalous points: " + str(df['label'].sum()))
    df_train = df.sample(frac=0.8)
    df_test = df.loc[~df.index.isin(df_train.index)]
    df_valid = df_train.sample(frac=0.25)
    
    print("Anomaly rate in test: " + str(df_test['label'].sum()/len(df_test.index)))

    df_train.drop(df_train[df_train.label == 1].index, inplace=True)
    df_valid.drop(df_valid[df_valid.label == 1].index, inplace=True)

    if os.path.exists("data/unswnb15/UNSW-NB15_training.csv"):
        os.remove("data/unswnb15/UNSW-NB15_training.csv")
    if os.path.exists("data/unswnb15/UNSW-NB15_test.csv"):
        os.remove("data/unswnb15/UNSW-NB15_test.csv")
    if os.path.exists("data/unswnb15/UNSW-NB15_valid.csv"):
        os.remove("data/unswnb15/UNSW-NB15_valid.csv")

    df_train.to_csv("data/unswnb15/UNSW-NB15_training.csv", index=False, header=True)
    df_test.to_csv("data/unswnb15/UNSW-NB15_test.csv", index=False, header=True)
    df_valid.to_csv("data/unswnb15/UNSW-NB15_valid.csv", index=False, header=True)

    # x_train, y_train = _to_xy(df_train, target='label')
    # x_valid, y_valid = _to_xy(df_valid, target='label')
    # x_test, y_test = _to_xy(df_test, target='label')
    #
    # y_test = y_test.flatten().astype(np.int)
    # x_train = x_train[y_train != 1]
    # x_valid = x_valid[y_valid != 1]
    #
    # dataset = {}
    # dataset['x_train'] = x_train.astype(np.float32)
    # dataset['x_valid'] = x_valid.astype(np.float32)
    # dataset['x_test'] = x_test.astype(np.float32)
    # dataset['y_test'] = y_test.astype(np.int)

    #return dataset


def to_xy(df, target):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)

    dummies = df[target]
    #return df.as_matrix(result).astype(np.float32), dummies.as_matrix().flatten().astype(np.int)
    return df[result].to_numpy(dtype=np.float32), df[target].to_numpy(dtype=np.int)

def _col_names():
    """Column names of the dataframe"""
   #id,dur,proto,service,state,spkts,dpkts,sbytes,dbytes,rate,sttl,dttl,sload,dload,sloss,dloss,sinpkt,dinpkt,sjit,djit,swin,stcpb,dtcpb,dwin,tcprtt,synack,ackdat,smean,dmean,trans_depth,response_body_len,ct_srv_src,ct_state_ttl,ct_dst_ltm,ct_src_dport_ltm,ct_dst_sport_ltm,ct_dst_src_ltm,is_ftp_login,ct_ftp_cmd,ct_flw_http_mthd,ct_src_ltm,ct_srv_dst,is_sm_ips_ports,attack_cat,label
    return ['srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss',
            'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth',
            'res_bdy_len', 'Sjit', 'Djit', 'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
            'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ ltm',
            'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label']


if __name__ == "__main__":
    _prepare_dataset()
