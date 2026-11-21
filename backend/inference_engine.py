import base64
import io
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


class GradCAMHook:
    """Captures forward activations and backward gradients from the final conv block."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.activations = None
        self.gradients = None
        self.fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out

    def _save_gradient(self, module, grad_inp, grad_out):
        self.gradients = grad_out[0]

    def generate(self, logits: torch.Tensor, class_idx: int, size=(224, 224)) -> np.ndarray:
        score = logits[0, class_idx]
        score.backward(retain_graph=True)

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam = cam / cam.max()
        return cv2.resize(cam, size)

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


class MultiModalScreeningEngine:
    def __init__(
        self,
        dr_weights="weights/resnet50_retinopathy_final.pth",
        dfu_weights="weights/best_dfu_model.pth",
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 1. Practical 2 Model: Diabetic Retinopathy (ResNet-50)
        self.dr_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        in_features_dr = self.dr_model.fc.in_features
        self.dr_loaded = os.path.exists(dr_weights)

        if self.dr_loaded:
            dr_state = torch.load(dr_weights, map_location=self.device, weights_only=True)
            dr_state = {k.replace("backbone.", "").replace("model.", ""): v for k, v in dr_state.items()}

            if "fc.0.weight" in dr_state and "fc.4.weight" in dr_state:
                hidden_dim = dr_state["fc.0.weight"].shape[0]
                num_dr_classes = dr_state["fc.4.weight"].shape[0]
                self.dr_model.fc = nn.Sequential(
                    nn.Linear(in_features_dr, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.3),
                    nn.Linear(hidden_dim, num_dr_classes),
                )
            else:
                self.dr_model.fc = nn.Linear(in_features_dr, 5)

            self.dr_model.load_state_dict(dr_state)
        else:
            self.dr_model.fc = nn.Linear(in_features_dr, 5)

        self.dr_model.to(self.device).eval()

        # 2. Practical 3 Model: Diabetic Foot Ulcer (EfficientNet-V2-S, 2 classes)
        self.dfu_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        in_features_dfu = self.dfu_model.classifier[1].in_features
        self.dfu_model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features_dfu, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 2),
        )
        self.dfu_loaded = os.path.exists(dfu_weights)
        if self.dfu_loaded:
            dfu_state = torch.load(dfu_weights, map_location=self.device, weights_only=True)
            dfu_state = {k.replace("backbone.", "").replace("model.", ""): v for k, v in dfu_state.items()}
            self.dfu_model.load_state_dict(dfu_state)
        self.dfu_model.to(self.device).eval()

    def _encode_overlay(self, raw_rgb: np.ndarray, heatmap: np.ndarray) -> str:
        colored_map = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        colored_map = cv2.cvtColor(colored_map, cv2.COLOR_BGR2RGB)
        blended = cv2.addWeighted(raw_rgb, 0.55, colored_map, 0.45, 0)
        pil_img = Image.fromarray(blended)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def analyze_retina(self, image_bytes: bytes) -> dict:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        raw_rgb = np.array(pil_img.resize((224, 224)))
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        tensor.requires_grad = True

        cam_hook = GradCAMHook(self.dr_model, self.dr_model.layer4[-1])
        logits = self.dr_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_grade = int(torch.argmax(probs).item())

        num_classes = probs.shape[0]
        if num_classes == 5:
            weights = torch.tensor([0.0, 0.25, 0.55, 0.85, 1.0], device=self.device)
            risk_score = float(torch.sum(probs * weights).item() * 100)
            grades = [
                "No DR (Grade 0)",
                "Mild NPDR (Grade 1)",
                "Moderate NPDR (Grade 2)",
                "Severe NPDR (Grade 3)",
                "Proliferative DR (Grade 4)",
            ]
            diagnosis_label = grades[pred_grade]
        else:
            risk_score = float(probs[1].item() * 100) if num_classes == 2 else float(probs[pred_grade].item() * 100)
            diagnosis_label = f"DR Stage {pred_grade}"

        heatmap = cam_hook.generate(logits, pred_grade)
        cam_hook.close()

        return {
            "diagnosis": diagnosis_label,
            "confidence": round(float(probs[pred_grade].item() * 100), 1),
            "risk_score": round(risk_score, 1),
            "gradcam_base64": self._encode_overlay(raw_rgb, heatmap),
        }

    def analyze_foot(self, image_bytes: bytes) -> dict:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        raw_rgb = np.array(pil_img.resize((224, 224)))
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)
        tensor.requires_grad = True

        cam_hook = GradCAMHook(self.dfu_model, self.dfu_model.features[-1])
        logits = self.dfu_model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())

        ulcer_risk = float(probs[0].item() * 100)
        heatmap = cam_hook.generate(logits, pred_idx)
        cam_hook.close()

        label = "Abnormal (Diabetic Foot Ulcer)" if pred_idx == 0 else "Normal (Healthy Plantar Skin)"
        return {
            "diagnosis": label,
            "confidence": round(float(probs[pred_idx].item() * 100), 1),
            "risk_score": round(ulcer_risk, 1),
            "gradcam_base64": self._encode_overlay(raw_rgb, heatmap),
        }
     