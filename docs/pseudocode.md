# PS-016 Algorithm Pseudocode

## 1. Baseline Algorithm: 3GPP Event A3 Handover

```text
Algorithm EventA3(ServingCell, RSRP_List, Hysteresis, TTT, dt):
    TargetCell = argmax(RSRP_List excluding ServingCell)
    
    if RSRP_List[TargetCell] > RSRP_List[ServingCell] + Hysteresis:
        TTT_Timer = TTT_Timer + dt
        if TTT_Timer >= TTT:
            TTT_Timer = 0
            return TRIGGER_HANDOVER, TargetCell
    else:
        TTT_Timer = 0
        
    return MAINTAIN_CONNECTION, ServingCell
```

## 2. Supervised Learning Handover Policy

```text
Algorithm SupervisedHandoverPolicy(ServingCell, RSRP_List, PrevRSRP_List, Speed, DwellTime, Classifier):
    TargetCell = argmax(RSRP_List excluding ServingCell)
    
    FeatureVector = [
        ServingRSRP    = RSRP_List[ServingCell],
        TargetRSRP     = RSRP_List[TargetCell],
        RSRPDiff       = RSRP_List[TargetCell] - RSRP_List[ServingCell],
        DeltaServing   = RSRP_List[ServingCell] - PrevRSRP_List[ServingCell],
        DeltaTarget    = RSRP_List[TargetCell] - PrevRSRP_List[TargetCell],
        MobilitySpeed  = Speed,
        CellDwellTime  = DwellTime
    ]
    
    Decision = Classifier.Predict(FeatureVector)
    
    if Decision == 1 and RSRP_List[TargetCell] > RSRP_List[ServingCell]:
        return TRIGGER_HANDOVER, TargetCell
        
    return MAINTAIN_CONNECTION, ServingCell
```
