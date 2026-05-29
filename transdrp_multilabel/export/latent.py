import pandas as pd
import torch
import torch.nn as nn
from transdrp_multilabel.contracts import OmicsTable

def extract_latent_table(
    model: nn.Module,  # AdversarialNetwork
    omics: OmicsTable,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    model.eval()
    x = omics.x.loc[list(omics.sample_ids)].values.astype("float32")
    rows = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            batch_x = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            # encoder is shared_encoder
            z = model.encoder(batch_x)
            if getattr(model.encoder, "norm_flag", False):
                z = torch.nn.functional.normalize(z, p=2, dim=1)
            z_np = z.cpu().numpy()
            for i, sid in enumerate(omics.sample_ids[start : start + batch_size]):
                row = {"sample_id": sid, "domain": omics.domain}
                for j in range(z_np.shape[1]):
                    row[f"latent_{j}"] = float(z_np[i, j])
                rows.append(row)
    return pd.DataFrame(rows)
