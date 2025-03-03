from azure.ai.evaluation.simulator import (
    AdversarialScenario,
    AdversarialSimulator,
    DirectAttackSimulator,
    Simulator,
)

from azure.ai.evaluation import (
    ProtectedMaterialEvaluator,
    ContentSafetyEvaluator,
    GroundednessProEvaluator,
    ECIEvaluator,
)

# No need to run harmfulContent Summarization RAI test

SIMULATOR_CONFIG = {
    "ip": {
        "simulator_class": AdversarialSimulator,
        "scenario": AdversarialScenario.ADVERSARIAL_CONTENT_PROTECTED_MATERIAL,
        "max_results": 250,  # 10,
        "max_conversation_turns": 1,
    },
    "eci": {
        "simulator_class": AdversarialSimulator,
        "scenario": _UnstableAdversarialScenario.ECI,
        "max_results": 250,  # 10,
        "max_conversation_turns": 1,
    },
    # Jailbreaking for upia
    "upia": {
        "simulator_class": DirectAttackSimulator,
        "scenario": AdversarialScenario.ADVERSARIAL_QA,
        "max_results": 200,  # 20,
        "max_conversation_turns": 1,
    },
    "grounding": {
        "simulator_class": Simulator,
        "scenario": None,
        "max_results": 287,  # 10,
        "max_conversation_turns": 1,
    },
    "safety": {
        "simulator_class": adversarial_simulator,
        "scenario": AdversarialScenario.ADVERSARIAL_QA,
        "max_results": 200,  # 20,
        "max_conversation_turns": 1,
    },
}


EVALUATOR_CLASSES = {
    "ip": ProtectedMaterialEvaluator,
    "upia": ContentSafetyEvaluator,
    "grounding": GroundednessProEvaluator,
    "eci": ECIEvaluator,
    "safety": ContentSafetyEvaluator,
}
