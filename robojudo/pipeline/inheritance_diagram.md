# `robojudo.pipeline` 类继承关系

下面的类图覆盖 `robojudo/pipeline/` 中定义的全部 class。箭头从父类指向子类；
`external` 表示父类定义在该目录之外。没有连线的类没有显式声明父类（隐式继承
`object`）。

```mermaid
classDiagram
    direction BT

    class ABC {
        <<external>>
    }
    class Config {
        <<external>>
    }
    class str {
        <<external>>
    }
    class Enum {
        <<external>>
    }

    class Pipeline {
        <<abstract>>
        base_pipeline.py
    }
    class RlPipeline {
        rl_pipeline.py
    }
    class RlMultiPolicyPipeline {
        rl_multi_policy_pipeline.py
    }
    class RlLocoMimicPipeline {
        rl_loco_mimic_pipeline.py
    }
    class G1LocomanipulationPipeline {
        g1_locomanipulation_pipeline.py
    }
    class G1LocomanipulationLocoMimicPipeline {
        g1_locomanipulation_loco_mimic_pipeline.py
    }
    class X2LocomanipulationPipeline {
        x2_locomanipulation_pipeline.py
    }
    class X2LocomanipulationLocoMimicPipeline {
        x2_locomanipulation_loco_mimic_pipeline.py
    }

    class UpperBodyZmqPipelineMixin {
        <<mixin>>
        upper_body_zmq_pipeline.py
    }
    class FourModePipelineMixin {
        <<mixin>>
        four_mode_pipeline.py
    }
    class G1FourModePipelineMixin {
        <<mixin>>
        g1_locomanipulation_pipeline.py
    }
    class X2FourModePipelineMixin {
        <<mixin>>
        x2_locomanipulation_pipeline.py
    }
    class LocomanipulationLocoMimicPipelineMixin {
        <<mixin>>
        locomanipulation_loco_mimic_pipeline.py
    }

    class PipelineCfg {
        <<config>>
        pipeline_cfgs.py
    }
    class RlPipelineCfg {
        <<config>>
        pipeline_cfgs.py
    }
    class RlMultiPolicyPipelineCfg {
        <<config>>
        pipeline_cfgs.py
    }
    class RlLocoMimicPipelineCfg {
        <<config>>
        pipeline_cfgs.py
    }

    class PolicyManager {
        <<helper>>
        rl_multi_policy_pipeline.py
    }
    class PolicyInterpManager {
        <<helper>>
        rl_loco_mimic_pipeline.py
    }
    class PolicyWrapper {
        <<helper>>
        rl_pipeline.py
    }
    class ControlMode {
        <<enumeration>>
        four_mode_pipeline.py
    }

    ABC <|-- Pipeline
    Pipeline <|-- RlPipeline
    RlPipeline <|-- RlMultiPolicyPipeline
    RlMultiPolicyPipeline <|-- RlLocoMimicPipeline

    UpperBodyZmqPipelineMixin <|-- FourModePipelineMixin
    FourModePipelineMixin <|-- G1FourModePipelineMixin
    FourModePipelineMixin <|-- X2FourModePipelineMixin

    G1FourModePipelineMixin <|-- G1LocomanipulationPipeline
    RlPipeline <|-- G1LocomanipulationPipeline

    LocomanipulationLocoMimicPipelineMixin <|-- G1LocomanipulationLocoMimicPipeline
    G1FourModePipelineMixin <|-- G1LocomanipulationLocoMimicPipeline
    RlLocoMimicPipeline <|-- G1LocomanipulationLocoMimicPipeline

    X2FourModePipelineMixin <|-- X2LocomanipulationPipeline
    RlPipeline <|-- X2LocomanipulationPipeline

    LocomanipulationLocoMimicPipelineMixin <|-- X2LocomanipulationLocoMimicPipeline
    X2FourModePipelineMixin <|-- X2LocomanipulationLocoMimicPipeline
    RlLocoMimicPipeline <|-- X2LocomanipulationLocoMimicPipeline

    Config <|-- PipelineCfg
    PipelineCfg <|-- RlPipelineCfg
    PipelineCfg <|-- RlMultiPolicyPipelineCfg
    PipelineCfg <|-- RlLocoMimicPipelineCfg

    PolicyManager <|-- PolicyInterpManager
    str <|-- ControlMode
    Enum <|-- ControlMode
```

## 阅读重点

- `RlPipeline -> RlMultiPolicyPipeline -> RlLocoMimicPipeline` 是运行时 Pipeline 的主继承链。
- G1 和 X2 的最终 Pipeline 使用多继承，把机器人/控制模式 Mixin 与运行时 Pipeline 组合起来。
- `LocomanipulationLocoMimicPipelineMixin` 没有自己的显式父类，但同时被 G1 和 X2 的 LocoMimic Pipeline 复用。
- 三个具体 RL 配置类都直接继承 `PipelineCfg`，它们之间没有继承关系。
- `PolicyWrapper` 是无显式父类的独立辅助类，因此图中没有继承连线。
