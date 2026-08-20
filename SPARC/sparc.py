                              

                      
                       
                                                                         
                                                             
                                                                   
                                                                            
   

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

                                                           
                                            
                                                           
class MPVA(nn.Module):
                          

    def __init__(self, in_dim, out_dim, rank, grid_size):
        super().__init__()
        self.rank = rank
        self.grid_size = grid_size
        self.reduction = nn.Linear(in_dim, rank, bias=False)
        self.local_context = nn.Conv2d(
            rank, rank, kernel_size=3, padding=1, groups=rank, bias=False
        )
        self.dilated_context = nn.Conv2d(
            rank, rank, kernel_size=3, padding=2, dilation=2,
            groups=rank, bias=False
        )
        self.expansion = nn.Linear(rank, out_dim, bias=False)
        self.residual_scale = 16.0 / rank
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.trunc_normal_(self.reduction.weight, std=0.01)
        nn.init.zeros_(self.local_context.weight)
        nn.init.zeros_(self.dilated_context.weight)
        nn.init.zeros_(self.expansion.weight)

    def forward(self, x):
        B, N, C_in = x.shape
        H, W = self.grid_size
        num_patches = H * W

        z = self.reduction(x)                

        if N >= num_patches:
            num_prompts = N - num_patches - 1
            z_cls = z[:, 0:1, :]                        
            z_patches = z[:, -num_patches:, :]                 

                     
            z_patches_2d = z_patches.transpose(1, 2).view(B, self.rank, H, W)
            z_sp = (
                z_patches_2d
                + self.local_context(z_patches_2d)
                + self.dilated_context(z_patches_2d)
            )

            if num_prompts > 0:
                z_prompts = z[:, 1: 1 + num_prompts, :]

                                                             
                z_sp_flat = z_sp.flatten(2).transpose(1, 2)                  
                scale = self.rank ** -0.5
                attn_scores = torch.matmul(z_cls, z_sp_flat.transpose(1, 2)) * scale              
                attn_weights = F.softmax(attn_scores, dim=-1)              
                visual_context = torch.matmul(attn_weights, z_sp_flat)

                                     
                z_prompts_interacted = z_prompts + visual_context

                                                      
                prompt_context = z_prompts.float().mean(
                    dim=1, keepdim=True
                ).to(z_prompts.dtype)
                prompt_context = prompt_context.transpose(1, 2).unsqueeze(-1)
                z_sp_interacted = z_sp + prompt_context
            else:
                z_prompts_interacted = None
                z_sp_interacted = z_sp

            z_patches_flat = z_sp_interacted.flatten(2).transpose(1, 2)
            if num_prompts > 0:
                z_recombined = torch.cat([z_cls, z_prompts_interacted, z_patches_flat], dim=1)
            else:
                z_recombined = torch.cat([z_cls, z_patches_flat], dim=1)

            return self.expansion(z_recombined) * self.residual_scale
        else:
            return self.expansion(z) * self.residual_scale


                                                           
                      
                                                           
class TaskSpecificMPVAProjection(nn.Module):
                                  

    def __init__(self, linear_layer, rank, grid_size):
        super().__init__()
        self.linear_layer = linear_layer
        self.rank = rank
        self.grid_size = grid_size
        for param in self.linear_layer.parameters():
            param.requires_grad = False
        self.task_mpva_modules = nn.ModuleList()
        self.active_task_id = 0

    def add_task_mpva(self):
        if self.rank == 0:
            return
        in_dim = self.linear_layer.weight.shape[1]
        out_dim = self.linear_layer.weight.shape[0]
        device = self.linear_layer.weight.device
        for mpva in self.task_mpva_modules:
            for param in mpva.parameters():
                param.requires_grad = False
        new_mpva = MPVA(in_dim, out_dim, self.rank, self.grid_size).to(device)
        self.task_mpva_modules.append(new_mpva)
        self.active_task_id = len(self.task_mpva_modules) - 1

    def forward(self, x):
        original_out = self.linear_layer(x)
        if (
            self.rank == 0
            or len(self.task_mpva_modules) == 0
            or self.active_task_id == -1
        ):
            return original_out
        current_mpva = self.task_mpva_modules[self.active_task_id]
        mpva_residual = current_mpva(x)
        return original_out + mpva_residual


                                                           
              
                                                           
class SPARC(nn.Module):
       
                    
                                          
                              
                              
       

    def __init__(self, backbone, num_prompts=10, rank=16):
        super().__init__()
        self.backbone = backbone
        self.num_prompts = num_prompts
        self.rank = rank
        self.embed_dim = backbone.embed_dim
        self.num_tasks_seen = 0
        self.active_task_id = 0

                            
        self.register_buffer(
            'frozen_class_prototypes', torch.zeros(100, self.embed_dim)
        )
        self.register_buffer('prototype_counts', torch.zeros(100))
        self.register_buffer(
            'class_task_ids', torch.zeros(100, dtype=torch.long)
        )
        self.current_task_class_offset = 0

                      
                                                                                   
        self.expert_prototypes_by_task = {}

                
        for param in self.backbone.parameters():
            param.requires_grad = False

               
        if hasattr(self.backbone, 'head'):
            if hasattr(self.backbone.head, 'heads'):
                for h in self.backbone.head.heads:
                    for param in h.parameters():
                        param.requires_grad = True
            else:
                for param in self.backbone.head.parameters():
                    param.requires_grad = True

        self.use_expert_prompts = num_prompts > 0
        if self.use_expert_prompts:
            self.expert_prompts = nn.ParameterList()
            print("[SPARC] SPEA and DPHC enabled.")

        grid_size = (14, 14)
        if rank > 0:
            for i, block in enumerate(self.backbone.blocks):
                if hasattr(block.attn, 'proj'):
                    block.attn.proj = TaskSpecificMPVAProjection(
                        block.attn.proj, rank, grid_size)
                if hasattr(block.mlp, 'fc2'):
                    block.mlp.fc2 = TaskSpecificMPVAProjection(
                        block.mlp.fc2, rank, grid_size)

                                                                        
                                 
                                                                        
    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.backbone.eval()
            for m in self.modules():
                if (
                    isinstance(m, TaskSpecificMPVAProjection)
                    and len(m.task_mpva_modules) > 0
                ):
                    m.task_mpva_modules.eval()
                    if m.active_task_id != -1:
                        m.task_mpva_modules[m.active_task_id].train(True)
            if hasattr(self.backbone, 'head'):
                self.backbone.head.train(True)
        return self

                                                                        
                
                                                                        
    def activate_task_branch(self, task_id):
        self.active_task_id = task_id
        for m in self.modules():
            if isinstance(m, TaskSpecificMPVAProjection):
                m.active_task_id = task_id

                                                                        
                               
                                                                        
    def add_task_branch(self, task_id, device='cuda'):
        self.num_tasks_seen = task_id + 1

        if task_id > 0 and hasattr(self.backbone.head, 'heads'):
            self.current_task_class_offset += \
                self.backbone.head.heads[task_id - 1].out_features

        if hasattr(self.backbone, 'head'):
            if hasattr(self.backbone.head, 'sigmas'):
                for s in self.backbone.head.sigmas:
                    s.requires_grad = False
            if hasattr(self.backbone.head, 'heads'):
                for h in self.backbone.head.heads[:-1]:
                    for param in h.parameters():
                        param.requires_grad = False
                for param in self.backbone.head.heads[-1].parameters():
                    param.requires_grad = True

        if self.rank > 0:
            for m in self.modules():
                if isinstance(m, TaskSpecificMPVAProjection):
                    m.add_task_mpva()

        if self.use_expert_prompts:
            for p in self.expert_prompts:
                p.requires_grad = False

            depth = len(self.backbone.blocks)
            new_prompt = nn.Parameter(
                torch.zeros(depth, self.num_prompts, self.embed_dim).to(device))
            nn.init.trunc_normal_(new_prompt, std=0.02)
            self.expert_prompts.append(new_prompt)

                                                                        
                                        
                                                                        
    def extract_frozen_features(self, x):
        previous_task_id = self.active_task_id
        self.activate_task_branch(-1)
        B = x.shape[0]
        x_embed = self.backbone.patch_embed(x)
        x_seq = torch.cat(
            (self.backbone.cls_token.expand(B, -1, -1), x_embed), dim=1
        ) + self.backbone.pos_embed
        x_seq = self.backbone.pos_drop(x_seq)
        with torch.no_grad():
            for block in self.backbone.blocks:
                x_seq = block(x_seq)
            deep_features = self.backbone.norm(x_seq)[:, 0]
        self.activate_task_branch(previous_task_id)
        return deep_features

                                                                        
                                          
                                                                        
    def forward_prompt_expert_features(self, x, task_id=None):
        if task_id is None:
            task_id = max(0, self.num_tasks_seen - 1)

        B = x.shape[0]
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.backbone.pos_embed
        x = self.backbone.pos_drop(x)

        if self.use_expert_prompts:
            task_prompts = self.expert_prompts[task_id]

            for i, block in enumerate(self.backbone.blocks):
                p_i = task_prompts[i].unsqueeze(0).expand(B, -1, -1)

                if i == 0:
                    x = torch.cat([x[:, :1, :], p_i, x[:, 1:, :]], dim=1)
                else:
                    x = torch.cat([x[:, :1, :], p_i, x[:, 1 + self.num_prompts:, :]], dim=1)

                x = block(x)
        else:
            for block in self.backbone.blocks:
                x = block(x)

        return self.backbone.norm(x)[:, 0]

                                                                        
                  
                                                                        
    @torch.no_grad()
    def construct_dual_space_prototypes(self, dataloader, device):
        self.eval()
        curr_task = self.num_tasks_seen - 1

                               
        for samples, targets in dataloader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            frozen_feat = self.extract_frozen_features(samples)
            frozen_feat = F.normalize(frozen_feat, p=2, dim=1)

            for i in range(len(targets)):
                c = targets[i].item()
                self.frozen_class_prototypes[c] += frozen_feat[i]
                self.prototype_counts[c] += 1
                self.class_task_ids[c] = curr_task

                 
        valid_classes_for_task = (
            self.class_task_ids == curr_task
        ).nonzero(as_tuple=True)[0]
        for c in valid_classes_for_task:
            if self.prototype_counts[c] > 0:
                self.frozen_class_prototypes[c] /= self.prototype_counts[c]

        print(f"[Prototype] Task {curr_task} Frozen Prototypes extracted. "
              f"Classes: {valid_classes_for_task.tolist()}")

                               
        self.activate_task_branch(curr_task)

        class_expert_feats = defaultdict(list)

        for samples, targets in dataloader:
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

                                               
            expert_feat = self.forward_prompt_expert_features(
                samples, task_id=curr_task
            )
            expert_feat = F.normalize(expert_feat, p=2, dim=1)

            for i in range(len(targets)):
                c = targets[i].item()
                class_expert_feats[c].append(expert_feat[i].cpu())           

                                                      
        proto_list = []
        class_order = sorted(class_expert_feats.keys())          
        for c in class_order:
            feats = torch.stack(class_expert_feats[c], dim=0)          
            proto = F.normalize(feats.mean(0, keepdim=True), p=2, dim=1)          
            proto_list.append(proto)

        if proto_list:
                                                          
            self.expert_prototypes_by_task[curr_task] = torch.cat(
                proto_list, dim=0
            )
            print(f"[Prototype] Task {curr_task} Expert Prototypes shape: "
                  f"{self.expert_prototypes_by_task[curr_task].shape}")
        else:
            print(f"[Prototype] Task {curr_task} WARNING: No expert prototypes extracted!")

                                                                        
                       
                                                                        
    def _compute_expert_relevance(self, expert_feat_t, task_id, device):
           
                                                         
                                   
           
        if task_id not in self.expert_prototypes_by_task:
                                          
            return torch.zeros(expert_feat_t.shape[0], device=device)

        protos = self.expert_prototypes_by_task[task_id].to(device)
        protos = F.normalize(protos, p=2, dim=1)
        feat_norm = F.normalize(expert_feat_t, p=2, dim=1)                 
        sim = torch.matmul(feat_norm, protos.T)                               
        return sim.max(dim=1)[0]                                         

                                                                        
                               
                                                                        
    def apply_ctcr_to_logits(self, logits):
        curr_task = self.num_tasks_seen - 1
        if curr_task == 0 or self.current_task_class_offset == 0:
            return logits

        current_class_mask = torch.ones_like(logits, dtype=torch.bool)
        current_class_mask[:, :self.current_task_class_offset] = False
        return torch.where(
            current_class_mask,
            logits,
            torch.tensor(-100.0, dtype=logits.dtype, device=logits.device),
        )

                                                                        
             
                                                                        
    def forward(self, x):
        if self.training:
            curr_task = self.num_tasks_seen - 1
            self.activate_task_branch(curr_task)
            features = self.forward_prompt_expert_features(
                x, task_id=curr_task
            )
            with torch.cuda.amp.autocast(enabled=False):
                logits = self.backbone.head(features.float())
            return self.apply_ctcr_to_logits(logits)

                                                                
        else:
                                    
            frozen_query = self.extract_frozen_features(x)
            valid_classes = (
                self.prototype_counts > 0
            ).nonzero(as_tuple=True)[0]
            missing_tasks = [
                t for t in range(self.num_tasks_seen)
                if not any(self.class_task_ids[valid_classes] == t)
            ]

            use_dphc = False

            if missing_tasks:
                curr_task = missing_tasks[-1]
                pred_task_ids = torch.full(
                    (x.shape[0],), curr_task, dtype=torch.long, device=x.device)
            elif len(valid_classes) == 0:
                curr_task = max(0, self.num_tasks_seen - 1)
                pred_task_ids = torch.zeros(
                    x.shape[0], dtype=torch.long, device=x.device)
            else:
                use_dphc = True

                                         
                valid_protos = F.normalize(
                    self.frozen_class_prototypes[valid_classes], p=2, dim=1)
                q_norm = F.normalize(frozen_query, p=2, dim=1, eps=1e-6)
                sim_frozen_all = torch.matmul(q_norm, valid_protos.T)                  

                             
                frozen_task_relevance = torch.zeros(
                    x.shape[0], self.num_tasks_seen, device=x.device)
                for t in range(self.num_tasks_seen):
                    t_mask = (self.class_task_ids[valid_classes] == t)
                    if t_mask.sum() > 0:
                        frozen_task_relevance[:, t] = (
                            sim_frozen_all[:, t_mask].max(dim=1)[0]
                        )

                                                      
            all_logits = []
                                       
            task_expert_feats = {}                     

            for t in range(self.num_tasks_seen):
                self.activate_task_branch(t)
                f_t = self.forward_prompt_expert_features(x, task_id=t)
                task_expert_feats[t] = f_t             

                if hasattr(self.backbone.head, 'heads'):
                    w_fp32 = self.backbone.head.heads[t].weight.float()
                    f_fp32 = f_t.float()
                    w_norm = F.normalize(w_fp32, p=2, dim=1, eps=1e-8).to(f_t.dtype)
                    f_norm_cls = F.normalize(f_fp32, p=2, dim=1, eps=1e-8).to(f_t.dtype)
                    cos_expert = F.linear(f_norm_cls, w_norm)

                    if use_dphc:
                        logits_t = cos_expert * 12.0
                    else:
                        if hasattr(self.backbone.head, 'sigmas'):
                            logits_t = cos_expert * self.backbone.head.sigmas[t].abs()
                        else:
                            logits_t = cos_expert * 12.0
                else:
                    logits_t = self.backbone.head(f_t)

                if not use_dphc:
                    mask = (pred_task_ids == t).unsqueeze(1)
                    logits_t = torch.where(
                        mask, logits_t,
                        torch.tensor(-1e3, device=x.device))

                all_logits.append(logits_t)

            self.activate_task_branch(max(0, self.num_tasks_seen - 1))

                                                                    
            if use_dphc:

                                     
                expert_task_relevance = torch.zeros(
                    x.shape[0], self.num_tasks_seen, device=x.device)

                for t in range(self.num_tasks_seen):
                    expert_relevance_t = self._compute_expert_relevance(
                        task_expert_feats[t], t, x.device)
                    expert_task_relevance[:, t] = expert_relevance_t

                                                    
                num_tasks_with_expert_proto = len(
                    self.expert_prototypes_by_task
                )
                if num_tasks_with_expert_proto == self.num_tasks_seen:
                    beta = 0.4
                elif num_tasks_with_expert_proto > 0:
                                         
                    beta = 0.4 * (
                        num_tasks_with_expert_proto / self.num_tasks_seen
                    )
                else:
                                        
                    beta = 0.0

                fused_task_relevance = (
                    (1.0 - beta) * frozen_task_relevance
                    + beta * expert_task_relevance
                )

                                                         
                tau = 12.0
                task_relevance_probs = F.softmax(
                    fused_task_relevance * tau, dim=1
                )
                task_log_priors = torch.log(
                    torch.clamp(task_relevance_probs, min=1e-6)
                )

                for t in range(self.num_tasks_seen):
                    all_logits[t] = (
                        all_logits[t] + task_log_priors[:, t:t + 1]
                    )

            return torch.cat(all_logits, dim=1)

                                                                        
          
                                                                        
    def expand_classifier(self, new_classes, device):
        if hasattr(self.backbone, 'increment_classes'):
            self.backbone.increment_classes(new_classes, device)

    @property
    def num_classes(self):
        if hasattr(self.backbone, 'head'):
            return self.backbone.head.out_features
        return 0

    @property
    def head(self):
        return self.backbone.head

    @property
    def pos_embed(self):
        return self.backbone.pos_embed
