import torch

class Tokenizer:
    def __init__(self, args):
        self.args = args
    
    def process_data(self, data_batch):
        data = data_batch[0].to(self.args.device, non_blocking=True)
        data = data[:, 0:2, :, :].to(torch.float32)
        
        if self.args.branch == "SPARTA_F":
            batch_size, channels, seg_lenx2, num_kp = data.shape
            seg_len = seg_lenx2 // 2
            input_data = data[:, :, :seg_len, :]
            target_data = data[:, :, seg_len:, :]
            
            input_data = self._apply_token_config(input_data, batch_size, channels, seg_len, num_kp)
            target_data = self._apply_token_config(target_data, batch_size, channels, seg_len, num_kp)
            
            return input_data, target_data
        else:
            batch_size, channels, seg_len, num_kp = data.shape
            data = self._apply_token_config(data, batch_size, channels, seg_len, num_kp)
            return data, data

    def _apply_token_config(self, data, batch_size, channels, seg_len, num_kp):
        if self.args.token_config == "t":
            data = data.permute(0, 2, 1, 3).reshape(batch_size, seg_len, channels * num_kp)
        elif self.args.token_config == "pst":
            data = data.reshape(batch_size, seg_len, channels * num_kp)
        elif self.args.token_config == "kps":
            data = self._process_kps(data, batch_size, seg_len, num_kp)
        elif self.args.token_config == "2ds":
            data = self._process_2ds(data, batch_size, seg_len, num_kp)
        elif self.args.token_config == "st":
            data = self._process_st(data, batch_size, channels, num_kp)
        return data
    
    def _process_kps(self, data, batch_size, seg_len, num_kp):
        if num_kp == 36:
            absolute, rel = data[:, :, :, :18], data[:, :, :, 18:]
            data = torch.cat([absolute, rel], dim=1).view(batch_size, -1, 18).permute(0, 2, 1)
        else:
            data = data.view(batch_size, -1, 18).permute(0, 2, 1)
        return data
    
    def _process_2ds(self, data, batch_size, seg_len, num_kp):
        if num_kp == 36:
            absolute, rel = data[:, :, :, :18], data[:, :, :, 18:]
            absolute = absolute.permute(0, 2, 1, 3).reshape(batch_size, seg_len, -1).permute(0, 2, 1)
            rel = rel.permute(0, 2, 1, 3).reshape(batch_size, seg_len, -1).permute(0, 2, 1)
            data = torch.cat([absolute, rel], dim=2)
        else:
            data = data.permute(0, 2, 1, 3).reshape(batch_size, seg_len, -1).permute(0, 2, 1)
        return data
    
    def _process_st(self, data, batch_size, channels, num_kp):
        if num_kp == 36:
            absolute, rel = data[:, :, :, :18], data[:, :, :, 18:]
            absolute = absolute.permute(0, 1, 3, 2).reshape(batch_size, channels, -1).permute(0, 2, 1)
            rel = rel.permute(0, 1, 3, 2).reshape(batch_size, channels, -1).permute(0, 2, 1)
            data = torch.cat([absolute, rel], dim=2)
        else:
            data = data.permute(0, 1, 3, 2).reshape(batch_size, channels, -1).permute(0, 2, 1)
        return data
