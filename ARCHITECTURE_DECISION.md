# GEPA+ART+RULER Architecture Decision Record (ADR)

## Decision: Local GPU Research Prototype

**Date**: August 27, 2025  
**Status**: ACTIVE - Emergency Architectural Recovery  
**Decision Maker**: System Analysis + Sequential Thinking MCP

## Problem Statement

The project suffered from **architectural schizophrenia**: 
- Setup/config designed for local GPU fine-tuning (4x RTX 3060, LoRA, transformers)
- Main pipeline implemented API-based RL training (OpenPipe, LangGraph agents)
- Core sophisticated components (RULER, LangGraph) completely bypassed
- No end-to-end integration between components

**Result**: System was fundamentally broken and unrunnable.

## Decision

**PRIMARY PARADIGM: Local GPU Research Prototype**

### What This System IS:
- **Research prototype** demonstrating GEPA+ART+RULER integration
- **Local GPU training** using 4x RTX 3060 (48GB VRAM)
- **LoRA fine-tuning** of Qwen models on competitive programming
- **Real evaluation** using OJBench/DMOJ judge server
- **Simple but working** component integration

### What This System is NOT:
- Production API-based RL training service
- Cloud-based LangGraph agent platform  
- OpenPipe ART framework integration
- Scalable multi-user system

## Architecture

```
Input: OJBench Problems
    ↓
[GEPA] Prompt Optimization (genetic algorithm)
    ↓ 
[ART] Structured Reasoning (local Qwen model)
    ↓
[RULER] Error Analysis & Correction
    ↓
[Local Training] LoRA fine-tuning on successful solutions
    ↓
Output: Improved model performance on competitive programming
```

## Components Status

### KEEP (Core Pipeline):
- ✅ `src/evaluation/real_ojbench.py` - Real OJBench evaluation
- ✅ `src/models/qwen_local_interface.py` - Local model interface  
- ✅ `src/config/art_training_config.py` - GPU training configuration
- ✅ `src/gepa/evolution_engine.py` - Prompt optimization
- ✅ `src/art/art_solver.py` - Structured reasoning solver
- ✅ `src/ruler/ruler_analyzer.py` - Error analysis

### QUARANTINE (Incompatible):
- ⚠️ `src/langgraph_art/` - API-based agent system (different paradigm)
- ⚠️ `src/models/openrouter_interface.py` - API client (not for local training)
- ⚠️ OpenPipe ART framework dependencies
- ⚠️ LangGraph agent pipeline

### REMOVE (Dead Code):
- ❌ All mock interfaces (already removed)
- ❌ Duplicate evaluation systems
- ❌ Conflicting configuration systems

## Integration Points

1. **GEPA → ART**: Optimized prompt passed to reasoning solver
2. **ART → RULER**: Failed solutions analyzed for error patterns  
3. **RULER → Training**: Correction guidance improves training data
4. **Training → Evaluation**: Fine-tuned model tested on held-out problems

## Success Criteria

- [ ] End-to-end pipeline completes without crashes
- [ ] GEPA produces measurably better prompts vs baseline
- [ ] ART solver generates valid C++ solutions  
- [ ] RULER identifies and corrects common error patterns
- [ ] Local training improves model performance >25% vs baseline
- [ ] System runs reliably on 4x RTX 3060 hardware

## Implementation Priority

1. **EMERGENCY**: Fix syntax errors preventing basic imports
2. **Phase 1**: Minimal working skeleton (GEPA → ART → Evaluation)
3. **Phase 2**: Add RULER error correction feedback loop
4. **Phase 3**: Local LoRA fine-tuning integration  
5. **Phase 4**: End-to-end optimization and evaluation

## Rejected Alternatives

- **API-based RL Training**: Requires external service, doesn't use available GPU hardware
- **Hybrid Local+API**: Too complex, leads to integration failures
- **Complete Redesign**: Time-prohibitive, wastes existing working components

## Next Actions

1. Create minimal working skeleton script
2. Test each integration point individually  
3. Add systematic end-to-end testing
4. Document actual data flow vs intended

---

**This ADR represents the definitive architectural direction. All development must align with this decision.**