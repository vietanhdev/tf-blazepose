import tensorflow as tf
import numpy as np
from ..utils.heatmap_process import post_process_heatmap

class PCKS(tf.keras.callbacks.Callback):
    def on_train_begin(self, model, logs={}):
        self._data = []
        self.model = model

    def on_epoch_end(self, batch, logs={}):
        X_val, y_val = self.validation_data[0], self.validation_data[1]
        y_predict = np.asarray(self.model.predict(X_val))

        y_val = np.argmax(y_val, axis=1)
        y_predict = np.argmax(y_predict, axis=1)

        self._data.append({
            'val_rocauc': roc_auc_score(y_val, y_predict),
        })
        return

    def get_data(self):
        return self._data


def heatmap2coor(hp_preds, n_kps = 7, img_size=(225,225)):
    heatmaps = hp_preds[:,:n_kps]
    flatten_hm = heatmaps.reshape((heatmaps.shape[0], n_kps, -1))
    flat_vectx = hp_preds[:,n_kps:2*n_kps].reshape((heatmaps.shape[0], n_kps, -1))
    flat_vecty = hp_preds[:,2*n_kps:].reshape((heatmaps.shape[0], n_kps, -1))
    flat_max = torch.argmax(flatten_hm, dim=-1)
    max_mask = flatten_hm == torch.unsqueeze(torch.max(flatten_hm, dim=-1)[0], dim=-1)
    cxs = flat_max%(heatmaps.shape[-2])
    cys = flat_max//(heatmaps.shape[-2])
    ovxs = torch.sum(flat_vectx*max_mask, dim=-1)
    ovys = torch.sum(flat_vectx*max_mask, dim=-1)
    xs_p = (cxs*8+ovxs)/img_size[1]
    ys_p = (cys*8+ovys)/img_size[0]
    hp_preds = torch.stack([xs_p, ys_p], dim=-1)
    return hp_preds
class PCKS(nn.Module):
    def __init__(self, pb_type='detection', n_kps=7, img_size=(225,225), id_shouder=(3,5), thresh=0.4):
        super(PCKS, self).__init__()
        self.n_kps =n_kps
        self.pb_type = pb_type
        self.img_size = img_size
        self.sr = id_shouder[0]
        self.sl = id_shouder[1]
        self.thresh = 0.4
    def forward(self, pred, target):
        ova_len = len(pred)*n_kps
        if self.pb_type == 'regression':
            shouders_len = ((target[...,self.sr:self.sr+1]-target[...,self.sl:self.sl+1])**2 + (target[...,self.sr+self.n_kps:self.sr+self.n_kps+1]-target[...,self.sl+self.n_kps:self.sl+self.n_kps+1])**2)**0.5
            err = torch.abs(pred-target)
            err = (err[...,:self.n_kps]**2 + err[...,self.n_kps]**2)**0.5
            err = torch.sum(err < shouders_len*self.thresh)
        elif self.pb_type == 'detection':
            pred = heatmap2coor(pred, self.n_kps, self.img_size)
            target = heatmap2coor(target, self.n_kps, self.img_size)
            shouders_len = ((target[:,self.sr:self.sr+1,0]-target[:,self.sl:self.sl+1,0])**2 + (target[:,self.sr:self.sr+1,1]-target[:,self.sl:self.sl+1,1])**2)**0.5
            err = torch.abs(pred-target)
            err = (err[...,0]**2 + err[...,1]**2)**0.5
            err = torch.sum(err < shouders_len*self.thresh)
        else:
            return None
        return err/ova_len