import os
import cv2
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader

class AIMSMultimodalDataset(Dataset):
    def __init__(self, video_paths: list, target_seq_len: int = 256, target_audio_dim: int = 128):
        """
        Custom DataLoader for the AIMS DTU 2026 Mandate.
        Extracts a single sparse frame and processes the continuous audio track.
        """
        self.video_paths = video_paths
        self.target_seq_len = target_seq_len
        self.target_audio_dim = target_audio_dim
        
        # Audio feature extractor: Converts raw soundwaves into a Mel-Spectrogram
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_mels=target_audio_dim,
            n_fft=1024,
            hop_length=512
        )

    def __len__(self):
        return len(self.video_paths)

    def extract_sparse_frame(self, video_path: str) -> torch.Tensor:
        """Opens the video and extracts the exact middle frame."""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_idx = total_frames // 2
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            # Fallback to a zero-tensor if video is corrupted
            return torch.zeros(3, 224, 224)
            
        # Convert BGR (OpenCV format) to RGB, resize, and convert to Tensor
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (224, 224)) # Standard visual encoder size
        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
        return frame_tensor

    def extract_audio_sequence(self, video_path: str) -> torch.Tensor:
        """Extracts the audio track and converts it to a sequence of Mel features."""
        try:
            # torchaudio can extract audio directly from an mp4 file
            waveform, sample_rate = torchaudio.load(video_path, format="mp4")
            
            # Resample to 16kHz for consistency
            if sample_rate != 16000:
                resampler = T.Resample(sample_rate, 16000)
                waveform = resampler(waveform)
                
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
                
            # Create Spectrogram
            mel_spec = self.mel_spectrogram(waveform).squeeze(0) # [Dim, Seq_Len]
            mel_spec = mel_spec.transpose(0, 1) # Convert to [Seq_Len, Dim]
            
            # Interpolate timeline to perfectly match our architectural sequence length (256)
            mel_spec = mel_spec.unsqueeze(0).unsqueeze(0) # Add batch/channel dims for interpolation
            mel_spec = torch.nn.functional.interpolate(mel_spec, size=(self.target_seq_len, self.target_audio_dim), mode='bilinear')
            mel_spec = mel_spec.squeeze(0).squeeze(0)
            
            return mel_spec
            
        except Exception as e:
            # Fallback if the video has no audio track
            return torch.zeros(self.target_seq_len, self.target_audio_dim)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        
        visual_frame = self.extract_sparse_frame(video_path)
        audio_sequence = self.extract_audio_sequence(video_path)
        
        return visual_frame, audio_sequence