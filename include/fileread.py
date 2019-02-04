import numpy as np

def read_feat(feat_path):
    """
    Get feature of utterance dictionary by reading *.fbk or *.mfc

    Input: feature file path
    Output: {utterance_id: numpy feature matrix}
    """
    feat_dict=dict()
    with open(feat_path, 'r') as f:
        for line in f:
            line = line[:-1]
            utt_id = line.split(" ")[0]
            utt_mat = []
            for line in f:
                line = line[:-1]
                frame_str = line.split(" ")
                frame_vec = [ float(val) for val in frame_str if val not in [ ']', ''] ]
                utt_mat.append(frame_vec)
                if frame_str[-1] == ']':
                    break
            feat_dict[utt_id] = np.array(utt_mat).astype('float32')
    return feat_dict

def read_text(feat_path):
    """
    Get phone label of utterance dictionary by reading *.text

    Input: text file path
    Output: {utterance_id: numpy phone list}
    """
    text_dict=dict()
    with open(feat_path, 'r') as f:
        for line in f:
            line = line[:-1]
            utt_id = line.split(" ")[0]
            phone_list = line.split(" ")[1:]
            text_dict[utt_id] = phone_list
    return text_dict