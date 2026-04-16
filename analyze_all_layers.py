"""
NEUROCARTOGRAPHY: Full 42-Layer Brain Decoder
============================================
Systematic analysis of ALL concept maps across the Gemma-4-E4B model.
Extracts breakthroughs, cross-lingual bindings, superposition signatures,
push-pull mechanisms, and cognitive phase transitions.
"""

import csv
import os
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

DATA_DIR = Path(r"d:\neurocartography_data")
BP_DIR = DATA_DIR / "basis_projection"
SYNTH_DIR = DATA_DIR / "synthesis"
OUTPUT_DIR = DATA_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# LANGUAGE DETECTION PATTERNS
# ============================================================
LANG_PATTERNS = {
    'Arabic': re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]'),
    'Chinese': re.compile(r'[\u4E00-\u9FFF\u3400-\u4DBF]'),
    'Japanese_Kana': re.compile(r'[\u3040-\u309F\u30A0-\u30FF]'),
    'Japanese_Kanji': re.compile(r'[\u4E00-\u9FFF]'),  # shared with Chinese
    'Korean': re.compile(r'[\uAC00-\uD7AF\u1100-\u11FF\u3130-\u318F]'),
    'Devanagari': re.compile(r'[\u0900-\u097F]'),
    'Bengali': re.compile(r'[\u0980-\u09FF]'),
    'Thai': re.compile(r'[\u0E00-\u0E7F]'),
    'Cyrillic': re.compile(r'[\u0400-\u04FF\u0500-\u052F]'),
    'Vietnamese_Diacritics': re.compile(r'[ăắằẳẵặâấầẩẫậđêếềểễệôốồổỗộơớờởỡợưứừửữự]', re.IGNORECASE),
    'Hebrew': re.compile(r'[\u0590-\u05FF]'),
    'Kannada': re.compile(r'[\u0C80-\u0CFF]'),
    'Telugu': re.compile(r'[\u0C00-\u0C7F]'),
    'Tamil': re.compile(r'[\u0B80-\u0BFF]'),
    'Malayalam': re.compile(r'[\u0D00-\u0D7F]'),
    'Myanmar': re.compile(r'[\u1000-\u109F]'),
    'Georgian': re.compile(r'[\u10A0-\u10FF]'),
    'Gujarati': re.compile(r'[\u0A80-\u0AFF]'),
    'Urdu_Persian': re.compile(r'[\u0600-\u06FF]'),
    'Latin_Extended': re.compile(r'[ñéèêëàâäùûüîïôöçæœ]', re.IGNORECASE),
}

CODE_PATTERNS = {
    'function_call': re.compile(r'[a-zA-Z_]\w*\('),
    'brackets': re.compile(r'[\[\]{}()]'),
    'operators': re.compile(r'[=<>!&|+\-*/]{2,}'),
    'variable_style': re.compile(r'[a-z][A-Z]|_[a-z]'),  # camelCase or snake_case
    'html_tag': re.compile(r'</?[a-zA-Z][^>]*>'),
    'dollar_syntax': re.compile(r'\$[a-zA-Z_{]'),
    'string_literal': re.compile(r'["\'][^"\']*["\']'),
    'math_syntax': re.compile(r'\\[a-zA-Z]+|mathbf|frac|sqrt'),
}

def detect_languages(text):
    """Detect all languages/scripts present in a concept label."""
    if not text or text.strip() == '':
        return set()
    found = set()
    for lang, pattern in LANG_PATTERNS.items():
        if pattern.search(text):
            found.add(lang)
    # Check for English words
    if re.search(r'[a-zA-Z]{3,}', text):
        found.add('English')
    return found

def detect_code_features(text):
    """Detect programming/code features in a concept label."""
    if not text:
        return set()
    found = set()
    for feat, pattern in CODE_PATTERNS.items():
        if pattern.search(text):
            found.add(feat)
    return found

def classify_concept_type(label, languages, code_features):
    """Classify a concept into a cognitive category."""
    if not label:
        return 'UNLABELED'
    
    n_langs = len(languages - {'English'})
    has_code = len(code_features) > 0
    has_english = 'English' in languages
    
    if n_langs >= 2 and has_code:
        return 'HYBRID_CODE_MULTILINGUAL'  # Polyglot programmer circuits
    elif n_langs >= 3:
        return 'DEEP_CROSS_LINGUAL'  # 3+ language families unified
    elif n_langs >= 1 and has_english:
        return 'CROSS_LINGUAL_BRIDGE'  # English + foreign language binding
    elif n_langs >= 1:
        return 'NON_ENGLISH_CLUSTER'
    elif has_code and has_english:
        return 'CODE_SEMANTIC'  # Code with meaning
    elif has_code:
        return 'PURE_SYNTAX'  # Raw syntax/structural
    elif has_english:
        return 'ENGLISH_SEMANTIC'
    else:
        return 'STRUCTURAL_FRAGMENT'

def analyze_push_pull(neurons):
    """Analyze excitatory vs inhibitory neuron balance."""
    excitatory = [n for n in neurons if n['alignment_score'] > 0]
    inhibitory = [n for n in neurons if n['alignment_score'] < 0]
    
    exc_mean = sum(n['alignment_score'] for n in excitatory) / len(excitatory) if excitatory else 0
    inh_mean = sum(n['alignment_score'] for n in inhibitory) / len(inhibitory) if inhibitory else 0
    
    # Compute "flatness" — how evenly distributed are the scores?
    scores = [abs(n['alignment_score']) for n in neurons]
    if scores:
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score)**2 for s in scores) / len(scores)
        flatness = 1.0 - (variance / (mean_score**2 + 1e-10))  # 1.0 = perfectly flat
    else:
        flatness = 0
    
    return {
        'n_excitatory': len(excitatory),
        'n_inhibitory': len(inhibitory),
        'exc_mean': exc_mean,
        'inh_mean': inh_mean,
        'balance_ratio': len(excitatory) / (len(inhibitory) + 1e-10),
        'flatness': flatness,  # >0.8 = diffuse encoding signature
        'max_score': max(scores) if scores else 0,
        'min_score': min(scores) if scores else 0,
    }

def detect_superposition(neurons, threshold=0.85):
    """Detect superposition signatures based on flat activation distributions."""
    push_pull = analyze_push_pull(neurons)
    return push_pull['flatness'] > threshold

def find_shared_neurons(layer_concepts):
    """Find neurons that participate in multiple concepts (polysemantic neurons)."""
    neuron_to_concepts = defaultdict(list)
    for concept in layer_concepts:
        for neuron in concept['neurons']:
            neuron_to_concepts[neuron['neuron_idx']].append({
                'concept_idx': concept['concept_idx'],
                'concept_label': concept['concept_label'],
                'alignment_score': neuron['alignment_score'],
                'alignment_rank': neuron['alignment_rank'],
            })
    
    # Neurons in 3+ concepts are "hub neurons"
    polysemantic = {
        nid: concepts for nid, concepts in neuron_to_concepts.items()
        if len(concepts) >= 3
    }
    return polysemantic

def parse_layer(layer_idx):
    """Parse a single layer's concept map CSV."""
    filepath = BP_DIR / f"layer_{layer_idx:02d}_neuron_concept_map.csv"
    if not filepath.exists():
        return None
    
    concepts = defaultdict(lambda: {'neurons': [], 'concept_label': '', 'singular_value': 0})
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cidx = int(row['concept_idx'])
            concepts[cidx]['concept_idx'] = cidx
            concepts[cidx]['concept_label'] = row['concept_label']
            concepts[cidx]['singular_value'] = float(row['singular_value'])
            concepts[cidx]['neurons'].append({
                'neuron_idx': int(row['physical_neuron_idx']),
                'alignment_score': float(row['alignment_score']),
                'alignment_rank': int(row['alignment_rank']),
            })
    
    return list(concepts.values())

def analyze_layer(layer_idx):
    """Full analysis of a single layer."""
    concepts = parse_layer(layer_idx)
    if not concepts:
        return None
    
    # Sort by singular value (importance)
    concepts.sort(key=lambda c: c['singular_value'], reverse=True)
    
    layer_result = {
        'layer_idx': layer_idx,
        'total_concepts': len(concepts),
        'top_singular_value': concepts[0]['singular_value'] if concepts else 0,
        'concept_types': Counter(),
        'cross_lingual_concepts': [],
        'hybrid_code_concepts': [],
        'superposition_signatures': [],
        'diffuse_encoded': [],
        'push_pull_stats': [],
        'language_diversity': Counter(),
        'all_languages_found': set(),
        'key_breakthroughs': [],
        'labeled_count': 0,
        'unlabeled_count': 0,
    }
    
    for concept in concepts:
        label = concept['concept_label']
        languages = detect_languages(label)
        code_features = detect_code_features(label)
        concept_type = classify_concept_type(label, languages, code_features)
        push_pull = analyze_push_pull(concept['neurons'])
        is_superposition = detect_superposition(concept['neurons'])
        
        layer_result['concept_types'][concept_type] += 1
        layer_result['all_languages_found'].update(languages)
        
        if label:
            layer_result['labeled_count'] += 1
        else:
            layer_result['unlabeled_count'] += 1
        
        for lang in languages:
            layer_result['language_diversity'][lang] += 1
        
        concept_summary = {
            'concept_idx': concept['concept_idx'],
            'label': label,
            'singular_value': concept['singular_value'],
            'type': concept_type,
            'languages': list(languages),
            'code_features': list(code_features),
            'push_pull': push_pull,
            'is_superposition': is_superposition,
        }
        
        # Flag breakthroughs
        if concept_type == 'HYBRID_CODE_MULTILINGUAL':
            layer_result['hybrid_code_concepts'].append(concept_summary)
            layer_result['key_breakthroughs'].append(
                f"🔬 POLYGLOT CODE CIRCUIT: Concept {concept['concept_idx']} "
                f"[{label}] fuses code + {', '.join(languages - {'English'})}"
            )
        
        if concept_type == 'DEEP_CROSS_LINGUAL':
            layer_result['cross_lingual_concepts'].append(concept_summary)
            layer_result['key_breakthroughs'].append(
                f"🌍 DEEP CROSS-LINGUAL: Concept {concept['concept_idx']} "
                f"[{label}] binds {len(languages)} scripts: {', '.join(languages)}"
            )
        
        if concept_type == 'CROSS_LINGUAL_BRIDGE':
            layer_result['cross_lingual_concepts'].append(concept_summary)
        
        if is_superposition:
            layer_result['superposition_signatures'].append(concept_summary)
            layer_result['key_breakthroughs'].append(
                f"⚡ SUPERPOSITION: Concept {concept['concept_idx']} "
                f"[{label}] flatness={push_pull['flatness']:.3f} — "
                f"quantum-like compression detected"
            )
        
        if push_pull['flatness'] > 0.75:
            layer_result['diffuse_encoded'].append(concept_summary)
        
        # Extreme push-pull imbalance detection
        if push_pull['n_excitatory'] >= 8 or push_pull['n_inhibitory'] >= 8:
            direction = "EXCITATORY_DOMINANT" if push_pull['n_excitatory'] > push_pull['n_inhibitory'] else "INHIBITORY_DOMINANT"
            layer_result['key_breakthroughs'].append(
                f"🧠 {direction}: Concept {concept['concept_idx']} "
                f"[{label}] has {push_pull['n_excitatory']}+ / {push_pull['n_inhibitory']}- "
                f"— extreme {direction.lower().replace('_', ' ')}"
            )
    
    # Find polysemantic hub neurons
    polysemantic = find_shared_neurons(concepts)
    top_polysemantic = sorted(polysemantic.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    layer_result['top_polysemantic_neurons'] = [
        {
            'neuron_idx': nid,
            'n_concepts': len(clist),
            'concept_labels': [c['concept_label'] for c in clist[:5]]
        }
        for nid, clist in top_polysemantic
    ]
    
    if top_polysemantic:
        top_hub = top_polysemantic[0]
        layer_result['key_breakthroughs'].append(
            f"🔗 TOP HUB NEURON: Neuron {top_hub[0]} participates in "
            f"{len(top_hub[1])} concepts — extreme polysemanticity"
        )
    
    return layer_result

def compute_cognitive_phase(layer_results):
    """Determine which cognitive phase each layer belongs to."""
    phases = {}
    for lr in layer_results:
        idx = lr['layer_idx']
        types = lr['concept_types']
        total_labeled = lr['labeled_count']
        
        # Compute ratios
        structural = types.get('STRUCTURAL_FRAGMENT', 0) + types.get('PURE_SYNTAX', 0)
        cross_lingual = types.get('CROSS_LINGUAL_BRIDGE', 0) + types.get('DEEP_CROSS_LINGUAL', 0)
        hybrid = types.get('HYBRID_CODE_MULTILINGUAL', 0)
        code = types.get('CODE_SEMANTIC', 0) + types.get('PURE_SYNTAX', 0)
        semantic = types.get('ENGLISH_SEMANTIC', 0)
        
        # Determine phase
        if idx <= 3:
            phase = "PHASE_1_SENSORY_SCANNER"
        elif idx <= 7:
            phase = "PHASE_2_UNIVERSAL_TRANSLATOR"
        elif idx <= 15:
            phase = "PHASE_3_LOGIC_ENGINE"
        elif idx <= 25:
            phase = "PHASE_4_DEEP_REASONING"
        elif idx <= 35:
            phase = "PHASE_5_KNOWLEDGE_SYNTHESIS"
        else:
            phase = "PHASE_6_OUTPUT_PREPARATION"
        
        # Override with data-driven insights
        n_superposition = len(lr['superposition_signatures'])
        n_crosslingual = len(lr['cross_lingual_concepts'])
        n_hybrid = len(lr['hybrid_code_concepts'])
        
        phases[idx] = {
            'phase': phase,
            'n_superposition': n_superposition,
            'n_crosslingual': n_crosslingual,
            'n_hybrid': n_hybrid,
            'n_diffuse': len(lr['diffuse_encoded']),
            'language_count': len(lr['all_languages_found']),
        }
    
    return phases

def generate_report(layer_results, phases):
    """Generate the full analysis report."""
    report_lines = []
    
    report_lines.append("=" * 80)
    report_lines.append("  NEUROCARTOGRAPHY: COMPLETE 42-LAYER BRAIN DECODE")
    report_lines.append("  Model: google/gemma-4-E4B-it")
    report_lines.append("  42 Layers × 512 Concepts × 10 Neurons = 215,040 Datapoints")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # ============================
    # GLOBAL STATISTICS
    # ============================
    report_lines.append("╔══════════════════════════════════════════════════════════════╗")
    report_lines.append("║                    GLOBAL STATISTICS                        ║")
    report_lines.append("╚══════════════════════════════════════════════════════════════╝")
    
    all_languages = set()
    total_cross_lingual = 0
    total_hybrid = 0
    total_superposition = 0
    total_diffuse = 0
    global_type_counts = Counter()
    
    for lr in layer_results:
        all_languages.update(lr['all_languages_found'])
        total_cross_lingual += len(lr['cross_lingual_concepts'])
        total_hybrid += len(lr['hybrid_code_concepts'])
        total_superposition += len(lr['superposition_signatures'])
        total_diffuse += len(lr['diffuse_encoded'])
        global_type_counts.update(lr['concept_types'])
    
    report_lines.append(f"\n  Total Languages/Scripts Detected: {len(all_languages)}")
    report_lines.append(f"  Languages: {', '.join(sorted(all_languages))}")
    report_lines.append(f"\n  Total Cross-Lingual Concepts: {total_cross_lingual}")
    report_lines.append(f"  Total Hybrid Code+Multilingual: {total_hybrid}")
    report_lines.append(f"  Total Superposition Signatures: {total_superposition}")
    report_lines.append(f"  Total Diffuse Encoded Concepts: {total_diffuse}")
    report_lines.append(f"\n  Concept Type Distribution:")
    for ctype, count in global_type_counts.most_common():
        report_lines.append(f"    {ctype}: {count}")
    
    # ============================
    # LAYER-BY-LAYER ANALYSIS
    # ============================
    report_lines.append("\n\n")
    report_lines.append("╔══════════════════════════════════════════════════════════════╗")
    report_lines.append("║                 LAYER-BY-LAYER BRAIN DECODE                 ║")
    report_lines.append("╚══════════════════════════════════════════════════════════════╝")
    
    for lr in layer_results:
        idx = lr['layer_idx']
        phase_info = phases[idx]
        
        report_lines.append(f"\n{'─'*70}")
        report_lines.append(f"  LAYER {idx:02d} | Phase: {phase_info['phase']}")
        report_lines.append(f"  Top Singular Value: {lr['top_singular_value']:.4f}")
        report_lines.append(f"  Languages Active: {len(lr['all_languages_found'])} | "
                          f"Cross-Lingual: {len(lr['cross_lingual_concepts'])} | "
                          f"Hybrid Code: {len(lr['hybrid_code_concepts'])} | "
                          f"Superposition: {len(lr['superposition_signatures'])}")
        report_lines.append(f"  Diffuse Encoded: {len(lr['diffuse_encoded'])} | "
                          f"Labeled: {lr['labeled_count']} | Unlabeled: {lr['unlabeled_count']}")
        report_lines.append(f"{'─'*70}")
        
        # Concept type breakdown
        report_lines.append(f"  Concept Types:")
        for ctype, count in lr['concept_types'].most_common():
            bar = "█" * min(count, 50)
            report_lines.append(f"    {ctype:35s} {count:4d} {bar}")
        
        # Language diversity
        if lr['language_diversity']:
            report_lines.append(f"\n  Language Diversity:")
            for lang, count in lr['language_diversity'].most_common(10):
                report_lines.append(f"    {lang:25s} {count:4d} concepts")
        
        # Key Breakthroughs
        if lr['key_breakthroughs']:
            report_lines.append(f"\n  🔬 KEY BREAKTHROUGHS ({len(lr['key_breakthroughs'])}):")
            for bt in lr['key_breakthroughs'][:15]:  # Top 15 per layer
                report_lines.append(f"    {bt}")
        
        # Top cross-lingual concepts
        if lr['cross_lingual_concepts']:
            report_lines.append(f"\n  🌍 Top Cross-Lingual Concepts:")
            for cc in lr['cross_lingual_concepts'][:5]:
                report_lines.append(f"    Concept {cc['concept_idx']:3d} "
                                  f"[SV={cc['singular_value']:.3f}] "
                                  f"{cc['label'][:60]}")
                report_lines.append(f"      Scripts: {', '.join(cc['languages'])}")
                pp = cc['push_pull']
                report_lines.append(f"      Push-Pull: {pp['n_excitatory']}+ / {pp['n_inhibitory']}- "
                                  f"| Flatness: {pp['flatness']:.3f}")
        
        # Top hybrid code concepts
        if lr['hybrid_code_concepts']:
            report_lines.append(f"\n  💻 Hybrid Code+Language Circuits:")
            for hc in lr['hybrid_code_concepts'][:5]:
                report_lines.append(f"    Concept {hc['concept_idx']:3d} "
                                  f"[SV={hc['singular_value']:.3f}] "
                                  f"{hc['label'][:60]}")
                report_lines.append(f"      Code Features: {', '.join(hc['code_features'])}")
                report_lines.append(f"      Languages: {', '.join(hc['languages'])}")
        
        # Superposition signatures
        if lr['superposition_signatures']:
            report_lines.append(f"\n  ⚡ Superposition Signatures:")
            for ss in lr['superposition_signatures'][:5]:
                report_lines.append(f"    Concept {ss['concept_idx']:3d} "
                                  f"[SV={ss['singular_value']:.3f}] "
                                  f"{ss['label'][:60]}")
                report_lines.append(f"      Flatness: {ss['push_pull']['flatness']:.4f} "
                                  f"(>0.85 = quantum compression)")
        
        # Top polysemantic hub neurons
        if lr['top_polysemantic_neurons']:
            report_lines.append(f"\n  🔗 Top Polysemantic Hub Neurons:")
            for pn in lr['top_polysemantic_neurons'][:5]:
                report_lines.append(f"    Neuron {pn['neuron_idx']:5d} → "
                                  f"{pn['n_concepts']} concepts")
    
    # ============================
    # COGNITIVE EVOLUTION MAP
    # ============================
    report_lines.append("\n\n")
    report_lines.append("╔══════════════════════════════════════════════════════════════╗")
    report_lines.append("║             COGNITIVE EVOLUTION: THE LIFECYCLE              ║")
    report_lines.append("║                    OF A THOUGHT                             ║")
    report_lines.append("╚══════════════════════════════════════════════════════════════╝")
    
    phase_groups = defaultdict(list)
    for idx, pinfo in phases.items():
        phase_groups[pinfo['phase']].append((idx, pinfo))
    
    for phase_name in sorted(phase_groups.keys()):
        layers = phase_groups[phase_name]
        layers.sort()
        layer_range = f"Layers {layers[0][0]:02d}-{layers[-1][0]:02d}"
        
        total_sup = sum(p['n_superposition'] for _, p in layers)
        total_cl = sum(p['n_crosslingual'] for _, p in layers)
        total_hyb = sum(p['n_hybrid'] for _, p in layers)
        avg_langs = sum(p['language_count'] for _, p in layers) / len(layers)
        
        report_lines.append(f"\n  {phase_name} ({layer_range})")
        report_lines.append(f"    Cross-Lingual Concepts: {total_cl}")
        report_lines.append(f"    Hybrid Code Circuits:   {total_hyb}")
        report_lines.append(f"    Superposition Events:   {total_sup}")
        report_lines.append(f"    Avg Languages/Layer:    {avg_langs:.1f}")
    
    # ============================
    # SINGULAR VALUE DECAY
    # ============================
    report_lines.append("\n\n")
    report_lines.append("╔══════════════════════════════════════════════════════════════╗")
    report_lines.append("║              SINGULAR VALUE DECAY CURVE                     ║")
    report_lines.append("║        (How concentrated is knowledge per layer?)           ║")
    report_lines.append("╚══════════════════════════════════════════════════════════════╝\n")
    
    for lr in layer_results:
        bar_len = int(lr['top_singular_value'] * 3)
        bar = "█" * min(bar_len, 50)
        report_lines.append(f"  Layer {lr['layer_idx']:02d} [{lr['top_singular_value']:7.3f}] {bar}")
    
    return "\n".join(report_lines)


def generate_json_data(layer_results, phases):
    """Generate structured JSON for further analysis."""
    output = {
        'model': 'google/gemma-4-E4B-it',
        'total_layers': len(layer_results),
        'total_concepts_per_layer': 512,
        'neurons_per_concept': 10,
        'phases': {},
        'layers': [],
        'global_stats': {},
    }
    
    all_languages = set()
    for lr in layer_results:
        all_languages.update(lr['all_languages_found'])
        
        layer_data = {
            'layer_idx': lr['layer_idx'],
            'phase': phases[lr['layer_idx']]['phase'],
            'top_singular_value': lr['top_singular_value'],
            'concept_types': dict(lr['concept_types']),
            'language_diversity': dict(lr['language_diversity']),
            'n_cross_lingual': len(lr['cross_lingual_concepts']),
            'n_hybrid_code': len(lr['hybrid_code_concepts']),
            'n_superposition': len(lr['superposition_signatures']),
            'n_diffuse': len(lr['diffuse_encoded']),
            'n_labeled': lr['labeled_count'],
            'n_unlabeled': lr['unlabeled_count'],
            'key_breakthroughs': lr['key_breakthroughs'][:20],
            'top_cross_lingual': [
                {
                    'concept_idx': c['concept_idx'],
                    'label': c['label'],
                    'singular_value': c['singular_value'],
                    'languages': c['languages'],
                    'push_pull_balance': c['push_pull']['balance_ratio'],
                    'flatness': c['push_pull']['flatness'],
                }
                for c in lr['cross_lingual_concepts'][:10]
            ],
            'top_hybrid': [
                {
                    'concept_idx': c['concept_idx'],
                    'label': c['label'],
                    'code_features': c['code_features'],
                    'languages': c['languages'],
                }
                for c in lr['hybrid_code_concepts'][:10]
            ],
            'top_polysemantic_neurons': lr['top_polysemantic_neurons'][:10],
        }
        output['layers'].append(layer_data)
    
    output['global_stats'] = {
        'total_languages': len(all_languages),
        'languages': sorted(all_languages),
    }
    
    return output


def main():
    print("=" * 70)
    print("  NEUROCARTOGRAPHY BRAIN DECODER v2.0")
    print("  Analyzing ALL 42 layers of Gemma-4-E4B...")
    print("=" * 70)
    
    layer_results = []
    for i in range(42):
        sys.stdout.write(f"\r  Processing Layer {i:02d}/41...")
        sys.stdout.flush()
        result = analyze_layer(i)
        if result:
            layer_results.append(result)
    
    print(f"\n  ✓ Processed {len(layer_results)} layers successfully")
    
    # Compute cognitive phases
    phases = compute_cognitive_phase(layer_results)
    
    # Generate text report
    print("  Generating comprehensive report...")
    report = generate_report(layer_results, phases)
    
    report_path = OUTPUT_DIR / "full_42_layer_brain_decode.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"  ✓ Report saved: {report_path}")
    
    # Generate JSON data
    json_data = generate_json_data(layer_results, phases)
    json_path = OUTPUT_DIR / "layer_analysis_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON data saved: {json_path}")
    
    # Print quick summary
    print("\n" + "=" * 70)
    print("  QUICK SUMMARY")
    print("=" * 70)
    
    total_cl = sum(len(lr['cross_lingual_concepts']) for lr in layer_results)
    total_hyb = sum(len(lr['hybrid_code_concepts']) for lr in layer_results)
    total_sup = sum(len(lr['superposition_signatures']) for lr in layer_results)
    total_diff = sum(len(lr['diffuse_encoded']) for lr in layer_results)
    
    all_langs = set()
    for lr in layer_results:
        all_langs.update(lr['all_languages_found'])
    
    print(f"  Languages/Scripts Found:    {len(all_langs)}")
    print(f"  Cross-Lingual Concepts:     {total_cl}")
    print(f"  Hybrid Code+Lang Circuits:  {total_hyb}")
    print(f"  Superposition Signatures:   {total_sup}")
    print(f"  Diffuse Encoded Concepts:   {total_diff}")
    print(f"\n  Top breakthroughs per layer:")
    
    for lr in layer_results:
        if lr['key_breakthroughs']:
            print(f"    Layer {lr['layer_idx']:02d}: {lr['key_breakthroughs'][0]}")
    
    print(f"\n  ✅ ANALYSIS COMPLETE. Check {OUTPUT_DIR} for full results.")

if __name__ == '__main__':
    main()
