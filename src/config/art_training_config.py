"""
ART training configuration with W&B integration and optimized hardware specs
Based on the blog post analysis for economical + fast balanced approach
"""
import os
from typing import Dict, Any, Optional
from pathlib import Path

class ARTTrainingConfig:
    """Configuration for ART RL training with GEPA+RULER integration"""
    
    def __init__(self):
        # Hardware Configuration (4x RTX 3060 Multi-GPU)
        self.hardware = {
            "gpu_type": "4x RTX 3060",  # Your actual setup
            "memory_gb_per_gpu": 12,
            "total_vram_gb": 48,  # 4 * 12GB
            "estimated_cost_per_hour": 0.032,  # Your actual rate
            "expected_training_time_hours": 12,  # Multi-GPU faster training
            "total_estimated_cost": 0.384  # $0.032 * 12h
        }
        
        # Alternative hardware options (for reference)
        self.hardware_alternatives = {
            "A100": {
                "cost_per_hour": 1.80,
                "training_time_hours": 12,
                "total_cost": 21.60
            },
            "H100": {
                "cost_per_hour": 2.50,
                "training_time_hours": 8,
                "total_cost": 20.00
            }
        }
        
        # Model Configuration - Qwen3-4B-Thinking-2507 (36GB base requirement)
        # Your setup: 4x RTX 3060 = 48GB total VRAM = 12GB headroom ✅
        self.model = {
            "name": "Qwen/Qwen3-4B-Thinking-2507",  # HuggingFace model path
            "local_path": "./models/qwen3-4b-thinking",  # Local download path
            "context_length": 4096,  # Conservative - thinking tokens are expensive
            "max_generation_tokens": 4096,  # Account for <think> block expansion
            "training_method": "LoRA",
            "precision": "fp16",
            "gradient_checkpointing": True,  # Essential for memory management
            "use_flash_attention": True,  # Memory + speed optimization
            "device_map": "auto",  # Auto-distribute across 4 GPUs (9GB per GPU)
            "torch_dtype": "float16",
            "load_in_8bit": False,  # Skip quantization - we have enough VRAM
            "low_cpu_mem_usage": True,  # Load directly to GPU
            "trust_remote_code": True,  # Required for Qwen thinking models
            "use_cache": False,  # Disable during training to save memory
            "lora_config": {
                "r": 16,  # Conservative rank to fit in remaining 12GB
                "lora_alpha": 32,
                "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
                "task_type": "CAUSAL_LM"
            }
        }
        
        # ART Training Parameters - Multi-GPU (4x RTX 3060, 48GB total)
        self.training = {
            "algorithm": "GRPO",  # Built into ART
            "epochs": 2,
            "learning_rate": 1.2e-5,
            "per_device_train_batch_size": 2,  # 2 per GPU = 8 total effective
            "per_device_eval_batch_size": 1,   # Conservative for evaluation
            "rollouts_per_prompt": 4,  # From blog post
            "gradient_accumulation_steps": 6,  # 8 * 6 = 48 effective batch size
            "warmup_steps": 100,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
            
            # Multi-GPU settings
            "ddp_backend": "nccl",  # Best for multi-GPU NVIDIA
            "dataloader_num_workers": 8,  # 2 per GPU
            "dataloader_pin_memory": True,  # We have enough system RAM
            
            # Checkpointing optimized for 4x GPU training
            "save_strategy": "steps",
            "save_steps": 25,  # More frequent saves for expensive training
            "save_total_limit": 5,  # Keep more checkpoints due to longer training
            "logging_steps": 5,   # More frequent logging
            "eval_steps": 50,     # More frequent evaluation
            "remove_unused_columns": False,
            "auto_resume": True,  # Critical for long multi-GPU training
            
            # Memory optimization for Qwen3-4B-Thinking
            "fp16": True,
            "fp16_full_eval": True,
            "gradient_checkpointing": True,
            "max_memory_mb": 45000,  # Reserve 3GB total across 4 GPUs
        }
        
        # RULER Reward Configuration
        self.rewards = {
            "success_reward": 4.0,
            "claude_judge_weight": 2.0,  # Higher weight for LLM judge
            "reasoning_quality": 1.5,
            "attempt_efficiency": 1.0,
            "time_efficiency": 1.0,
            "tool_usage": 0.8,
            "error_recovery": 1.2
        }
        
        # Checkpointing and Recovery Configuration
        self.checkpointing = {
            "checkpoint_dir": "./checkpoints",
            "save_total_limit": 3,  # Keep only 3 most recent checkpoints
            "resume_from_checkpoint": True,
            "auto_find_checkpoint": True,  # Automatically find latest checkpoint
            "push_to_hub": False,  # Set to True if using HF Hub
            "hub_model_id": None,
            "hub_strategy": "every_save"
        }
        
        # Memory Management for 24GB GPU
        self.memory_config = {
            "per_device_train_batch_size": 1,  # Very conservative
            "per_device_eval_batch_size": 1,
            "gradient_checkpointing": True,
            "dataloader_pin_memory": False,  # Reduce GPU memory pressure
            "max_memory_mb": 20000,  # Reserve 4GB for system
            "offload_optimizer_device": "cpu",  # Offload optimizer to CPU
            "offload_param_device": "cpu",  # Offload parameters when not in use
            "zero_stage": 2,  # ZeRO stage 2 for memory efficiency
            "fp16_full_eval": True  # Use FP16 for evaluation too
        }

        # W&B Configuration with Resume Support
        self.wandb = {
            "project": "gepa-art-ruler-qwen4b",
            "entity": os.getenv("WANDB_ENTITY", "competitive-programming-ai"),
            "tags": ["gepa", "art", "ruler", "qwen3-4b", "competitive-programming"],
            "resume": "allow",  # Allow resuming W&B runs
            "id": None,  # Will be set automatically or from checkpoint
            "config": {
                "model": self.model,
                "training": self.training,
                "hardware": self.hardware,
                "rewards": self.rewards,
                "checkpointing": self.checkpointing,
                "memory_config": self.memory_config
            }
        }
        
        # Evaluation Configuration
        self.evaluation = {
            "eval_frequency": 250,  # Every N training steps
            "eval_problems": 20,    # Small validation set
            "success_threshold": 0.40,  # Target >40% improvement
            "early_stopping_patience": 3
        }
        
        # Data Configuration
        self.data = {
            "train_problems": 100,  # Reasonable for 4B model
            "val_problems": 30,
            "test_problems": 50,
            "difficulty_distribution": {
                "easy": 0.4,
                "medium": 0.5, 
                "hard": 0.1
            }
        }
        
    @classmethod
    def get_wandb_config(cls) -> Dict[str, Any]:
        """Get W&B configuration"""
        config = cls()
        return {
            "project": config.wandb["project"],
            "entity": config.wandb["entity"],
            "tags": config.wandb["tags"],
            "config": config.wandb["config"]
        }
    
    @classmethod
    def get_art_trainer_config(cls) -> Dict[str, Any]:
        """Get configuration for ART trainer initialization"""
        config = cls()
        return {
            "model_config": config.model,
            "training_config": config.training,
            "reward_weights": config.rewards,
            "hardware_specs": config.hardware,
            "evaluation_config": config.evaluation
        }
    
    def estimate_total_cost(self, 
                           gepa_iterations: int = 3,
                           art_training_hours: Optional[float] = None) -> Dict[str, float]:
        """Estimate total training costs"""
        
        # GEPA costs (minimal - just API calls to Claude)
        gepa_cost = gepa_iterations * 0.50  # ~$0.50 per iteration with Claude API
        
        # ART training costs
        training_hours = art_training_hours or self.hardware["expected_training_time_hours"]
        art_cost = training_hours * self.hardware["estimated_cost_per_hour"]
        
        # W&B costs (free tier sufficient for this scale)
        wandb_cost = 0.0
        
        # OpenRouter API costs for Claude judge
        judge_cost = self.data["train_problems"] * 0.01  # ~$0.01 per trajectory evaluation
        
        total_cost = gepa_cost + art_cost + wandb_cost + judge_cost
        
        return {
            "gepa_preprocessing": gepa_cost,
            "art_training": art_cost,
            "claude_judge_scoring": judge_cost,
            "wandb_monitoring": wandb_cost,
            "total_estimated_cost": total_cost,
            "cost_breakdown": {
                "compute": art_cost,
                "apis": gepa_cost + judge_cost,
                "monitoring": wandb_cost
            }
        }
    
    def get_performance_targets(self) -> Dict[str, float]:
        """Get performance improvement targets"""
        return {
            "baseline_success_rate": 0.179,  # 17.9% from docs
            "gepa_boost_target": 0.25,       # 25% after GEPA optimization
            "final_target": 0.50,            # 50% after ART training
            "minimum_improvement": 0.40,     # >40% relative improvement
            "target_solve_time": 60.0        # Average seconds per problem
        }
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate the training configuration"""
        validation = {
            "valid": True,
            "warnings": [],
            "errors": []
        }
        
        # Check environment variables
        required_env_vars = ["OPENROUTER_API_KEY"]
        optional_env_vars = ["WANDB_API_KEY", "WANDB_ENTITY"]
        
        for var in required_env_vars:
            if not os.getenv(var):
                validation["errors"].append(f"Missing required environment variable: {var}")
                validation["valid"] = False
        
        for var in optional_env_vars:
            if not os.getenv(var):
                validation["warnings"].append(f"Optional environment variable not set: {var}")
        
        # Validate hardware requirements
        total_vram_gb = self.hardware.get("total_vram_gb", 0)
        if total_vram_gb < 20:
            validation["warnings"].append(f"Low GPU memory ({total_vram_gb}GB) may cause issues with 4B model + LoRA")
        
        # Validate batch size vs memory
        per_device_batch = self.training.get("per_device_train_batch_size", 1)
        gpu_count = 4  # Based on hardware config
        effective_batch = per_device_batch * gpu_count * self.training.get("gradient_accumulation_steps", 1)
        total_memory_usage = effective_batch * self.model["context_length"] * 2  # Rough estimate
        if total_memory_usage > 100000:  # Updated threshold for 48GB
            validation["warnings"].append(f"Effective batch size ({effective_batch}) may be too large for available memory")
        
        return validation


# W&B Integration Functions
def setup_wandb_logging():
    """Setup Weights & Biases logging"""
    try:
        import wandb
        
        config = ARTTrainingConfig.get_wandb_config()
        
        # Initialize W&B run
        wandb.init(
            project=config["project"],
            entity=config.get("entity"),
            tags=config["tags"],
            config=config["config"]
        )
        
        print("✅ W&B logging initialized")
        return True
        
    except ImportError:
        print("⚠️ W&B not available, install with: pip install wandb")
        return False
    except Exception as e:
        print(f"⚠️ W&B initialization failed: {str(e)}")
        return False

def log_training_metrics(metrics: Dict[str, Any], step: int):
    """Log metrics to W&B"""
    try:
        import wandb
        wandb.log(metrics, step=step)
    except:
        pass  # Fail silently if W&B not available

def log_trajectory_results(trajectories: list, prefix: str = "train"):
    """Log trajectory evaluation results to W&B"""
    try:
        import wandb
        
        success_rate = sum(1 for t in trajectories if t.success) / len(trajectories)
        avg_attempts = sum(t.attempt_number for t in trajectories) / len(trajectories)
        avg_claude_score = sum(getattr(t, 'claude_judge_score', 0.5) for t in trajectories) / len(trajectories)
        
        metrics = {
            f"{prefix}/success_rate": success_rate,
            f"{prefix}/avg_attempts": avg_attempts,
            f"{prefix}/avg_claude_score": avg_claude_score,
            f"{prefix}/total_trajectories": len(trajectories)
        }
        
        wandb.log(metrics)
        
    except:
        pass

def save_model_checkpoint(model_path: str, metrics: Dict[str, Any]):
    """Save model checkpoint as W&B artifact"""
    try:
        import wandb
        
        artifact = wandb.Artifact(
            name="qwen3-4b-checkpoint",
            type="model",
            metadata=metrics
        )
        
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)
        
        print(f"✅ Model checkpoint saved to W&B: {model_path}")
        
    except:
        pass


if __name__ == "__main__":
    # Test configuration
    config = ARTTrainingConfig()
    
    print("🔧 ART Training Configuration")
    print(f"Target Model: {config.model['name']}")
    print(f"Hardware: {config.hardware['gpu_type']} ({config.hardware['memory_gb']}GB)")
    print(f"Estimated Cost: ${config.hardware['total_estimated_cost']:.2f}")
    
    # Estimate costs
    cost_breakdown = config.estimate_total_cost()
    print(f"\n💰 Cost Breakdown:")
    for item, cost in cost_breakdown.items():
        if isinstance(cost, dict):
            continue
        print(f"  {item}: ${cost:.2f}")
    
    # Performance targets
    targets = config.get_performance_targets()
    print(f"\n🎯 Performance Targets:")
    print(f"  Baseline → GEPA: {targets['baseline_success_rate']:.1%} → {targets['gepa_boost_target']:.1%}")
    print(f"  Final Target: {targets['final_target']:.1%}")
    
    # Validate configuration
    validation = config.validate_configuration()
    print(f"\n✅ Configuration: {'Valid' if validation['valid'] else 'Invalid'}")
    for warning in validation["warnings"]:
        print(f"  ⚠️ {warning}")
    for error in validation["errors"]:
        print(f"  ❌ {error}")