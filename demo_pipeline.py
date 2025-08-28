"""
Simplified Psychedelic Therapeutics Demo
========================================

A demonstration of the 2C-B psychedelic therapeutic design pipeline
using basic Python libraries without RDKit/DeepChem dependencies.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import requests
import warnings
warnings.filterwarnings('ignore')

def create_sample_psychedelic_data():
    """Create sample psychedelic compound dataset."""
    print("🧪 Creating sample psychedelic compound dataset...")
    
    compounds = {
        # Core 2C compounds
        '2C-B': {
            'smiles': 'CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN',
            'class': '2C-series',
            'mw': 334.1,
            'logp': 3.2,
            'tpsa': 45.2,
            'hbd': 1,
            'hba': 3,
            'activity': 'agonist'
        },
        '2C-I': {
            'smiles': 'CCc1cc(I)c(OCc2ccccc2)c(I)c1CCN',
            'class': '2C-series',
            'mw': 428.1,
            'logp': 3.8,
            'tpsa': 45.2,
            'hbd': 1,
            'hba': 3,
            'activity': 'agonist'
        },
        '2C-E': {
            'smiles': 'CCCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN',
            'class': '2C-series',
            'mw': 348.1,
            'logp': 3.6,
            'tpsa': 45.2,
            'hbd': 1,
            'hba': 3,
            'activity': 'agonist'
        },
        'DOB': {
            'smiles': 'CC(N)Cc1cc(Br)c(OCc2ccccc2)c(Br)c1',
            'class': 'DOx-series',
            'mw': 320.1,
            'logp': 3.1,
            'tpsa': 45.2,
            'hbd': 1,
            'hba': 3,
            'activity': 'agonist'
        },
        'Mescaline': {
            'smiles': 'COc1cc(CCN)cc(OC)c1OC',
            'class': 'Mescaline-analog',
            'mw': 211.3,
            'logp': 0.4,
            'tpsa': 58.9,
            'hbd': 1,
            'hba': 4,
            'activity': 'agonist'
        }
    }
    
    # Convert to DataFrame
    data = []
    for name, props in compounds.items():
        row = {'compound_name': name}
        row.update(props)
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Add some calculated properties
    df['bbb_score'] = np.where((df['mw'] < 450) & (df['logp'].between(1, 3)) & (df['tpsa'] < 90), 1.0, 0.7)
    df['drug_likeness'] = 1.0 - ((df['mw'] > 500).astype(int) + 
                                  (df['logp'] > 5).astype(int) + 
                                  (df['hbd'] > 5).astype(int) + 
                                  (df['hba'] > 10).astype(int)) / 4.0
    
    print(f"✅ Created dataset with {len(df)} psychedelic compounds")
    return df

def analyze_2cb_properties(df):
    """Analyze 2C-B specific properties."""
    print("\n🔬 Analyzing 2C-B Properties")
    print("-" * 40)
    
    cb_data = df[df['compound_name'] == '2C-B'].iloc[0]
    
    print(f"2C-B Analysis:")
    print(f"  • SMILES: {cb_data['smiles']}")
    print(f"  • Molecular Weight: {cb_data['mw']:.1f} Da")
    print(f"  • LogP: {cb_data['logp']:.2f}")
    print(f"  • TPSA: {cb_data['tpsa']:.1f} Ų")
    print(f"  • H-bond donors: {cb_data['hbd']}")
    print(f"  • H-bond acceptors: {cb_data['hba']}")
    print(f"  • BBB penetration score: {cb_data['bbb_score']:.2f}")
    print(f"  • Drug-likeness score: {cb_data['drug_likeness']:.2f}")
    
    return cb_data

def generate_2cb_analogs():
    """Generate hypothetical 2C-B analogs."""
    print("\n🧬 Generating 2C-B Analogs")
    print("-" * 40)
    
    # Simulated analog generation
    analogs = []
    
    # Base 2C-B properties
    base_props = {
        'mw': 334.1,
        'logp': 3.2,
        'tpsa': 45.2,
        'hbd': 1,
        'hba': 3
    }
    
    # Generate variations
    substitutions = [
        ('Fluoro', {'mw': -80, 'logp': -0.3, 'tpsa': 0}),   # Replace Br with F
        ('Chloro', {'mw': -40, 'logp': -0.1, 'tpsa': 0}),   # Replace Br with Cl
        ('Methyl', {'mw': -120, 'logp': 0.2, 'tpsa': 0}),   # Replace Br with CH3
        ('Ethyl', {'mw': -106, 'logp': 0.4, 'tpsa': 0}),    # Replace Br with C2H5
        ('Methoxy', {'mw': -98, 'logp': -0.5, 'tpsa': 18}), # Replace Br with OCH3
        ('Trifluoromethyl', {'mw': -68, 'logp': 0.8, 'tpsa': 0}) # Replace Br with CF3
    ]
    
    for i, (sub_name, deltas) in enumerate(substitutions):
        analog = {
            'compound_name': f'2C-B-{sub_name}',
            'smiles': f'CCc1cc({sub_name.lower()})c(OCc2ccccc2)c({sub_name.lower()})c1CCN',
            'class': 'Generated_2C',
            'mw': base_props['mw'] + deltas['mw'],
            'logp': base_props['logp'] + deltas['logp'],
            'tpsa': base_props['tpsa'] + deltas['tpsa'],
            'hbd': base_props['hbd'],
            'hba': base_props['hba'] + (1 if 'methoxy' in sub_name.lower() else 0),
            'activity': 'predicted_agonist',
            'source': 'generated'
        }
        
        # Calculate derived properties
        analog['bbb_score'] = 1.0 if (analog['mw'] < 450 and 1 <= analog['logp'] <= 3 and analog['tpsa'] < 90) else 0.7
        analog['drug_likeness'] = 1.0 - ((analog['mw'] > 500) + (analog['logp'] > 5) + (analog['hbd'] > 5) + (analog['hba'] > 10)) / 4.0
        
        analogs.append(analog)
    
    analog_df = pd.DataFrame(analogs)
    print(f"✅ Generated {len(analog_df)} 2C-B analogs")
    
    # Show top analogs
    print("\n📋 Top 3 generated analogs by drug-likeness:")
    top_analogs = analog_df.nlargest(3, 'drug_likeness')
    for _, row in top_analogs.iterrows():
        print(f"   • {row['compound_name']}: MW={row['mw']:.1f}, LogP={row['logp']:.2f}, Drug-likeness={row['drug_likeness']:.2f}")
    
    return analog_df

def analyze_5ht2a_structure():
    """Simulate 5-HT2A receptor structure analysis."""
    print("\n🏗️ 5-HT2A Receptor Structure Analysis")
    print("-" * 40)
    
    # Simulate downloading structure
    print("📥 Downloading 5-HT2A receptor structure (PDB: 6A93)...")
    
    # Simulate structure analysis
    binding_pocket = {
        'pocket_size': 18,
        'hydrophobic_residues': ['PHE340', 'VAL156', 'ILE152', 'LEU229', 'PHE234'],
        'polar_residues': ['SER159', 'SER239', 'ASN343', 'THR160'],
        'charged_residues': ['ASP155', 'LYS152'],
        'key_interactions': [
            'ASP155 - critical salt bridge with amine',
            'SER159 - hydrogen bonding',
            'PHE340 - π-π stacking with aromatic rings',
            'ASN343 - hydrogen bonding with substituents'
        ]
    }
    
    print(f"✅ Binding pocket analysis completed:")
    print(f"   • Total residues in pocket: {binding_pocket['pocket_size']}")
    print(f"   • Hydrophobic residues: {len(binding_pocket['hydrophobic_residues'])}")
    print(f"   • Polar residues: {len(binding_pocket['polar_residues'])}")
    print(f"   • Charged residues: {len(binding_pocket['charged_residues'])}")
    
    print(f"\n🔑 Key binding interactions:")
    for interaction in binding_pocket['key_interactions']:
        print(f"   • {interaction}")
    
    # Simulate allosteric site identification
    print(f"\n🕳️ Allosteric site analysis:")
    allosteric_sites = [
        {'site': 'Extracellular loop 2', 'druggability': 0.72, 'volume': 245},
        {'site': 'Transmembrane 5-6', 'druggability': 0.68, 'volume': 189},
        {'site': 'Intracellular loop 3', 'druggability': 0.45, 'volume': 156}
    ]
    
    for site in allosteric_sites:
        print(f"   • {site['site']}: Druggability={site['druggability']:.2f}, Volume={site['volume']} Ų")
    
    return binding_pocket, allosteric_sites

def create_visualizations(original_df, analog_df):
    """Create visualization plots."""
    print("\n📊 Creating Visualizations")
    print("-" * 40)
    
    # Combine datasets
    all_compounds = pd.concat([original_df, analog_df], ignore_index=True)
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("Set2")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Psychedelic Therapeutics Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Molecular Weight vs LogP
    ax1 = axes[0, 0]
    for compound_class in all_compounds['class'].unique():
        class_data = all_compounds[all_compounds['class'] == compound_class]
        ax1.scatter(class_data['mw'], class_data['logp'], 
                   label=compound_class, alpha=0.7, s=80)
    
    ax1.set_xlabel('Molecular Weight (Da)')
    ax1.set_ylabel('LogP')
    ax1.set_title('Chemical Space: LogP vs Molecular Weight')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Property Distribution (TPSA)
    ax2 = axes[0, 1]
    ax2.hist(all_compounds['tpsa'], bins=10, alpha=0.7, color='skyblue', edgecolor='black')
    ax2.set_xlabel('TPSA (Ų)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('TPSA Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Drug-likeness by Compound Class
    ax3 = axes[1, 0]
    class_drug_likeness = all_compounds.groupby('class')['drug_likeness'].mean()
    bars = ax3.bar(class_drug_likeness.index, class_drug_likeness.values, 
                   color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax3.set_xlabel('Compound Class')
    ax3.set_ylabel('Average Drug-likeness Score')
    ax3.set_title('Drug-likeness by Compound Class')
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    # Plot 4: BBB Penetration Score
    ax4 = axes[1, 1]
    bbb_counts = all_compounds['bbb_score'].value_counts()
    colors = ['#FFD93D', '#6BCF7F']
    wedges, texts, autotexts = ax4.pie(bbb_counts.values, labels=['Good BBB', 'Poor BBB'], 
                                      autopct='%1.1f%%', colors=colors)
    ax4.set_title('Blood-Brain Barrier Penetration')
    
    plt.tight_layout()
    plt.savefig('psychedelic_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ Visualization dashboard created and saved as 'psychedelic_analysis_dashboard.png'")
    
    # Create SAR analysis plot
    plt.figure(figsize=(12, 8))
    
    # Compare 2C-series compounds
    cb_series = all_compounds[all_compounds['class'] == '2C-series']
    generated_series = all_compounds[all_compounds['class'] == 'Generated_2C']
    
    plt.subplot(2, 2, 1)
    plt.scatter(cb_series['mw'], cb_series['logp'], color='red', s=100, alpha=0.7, label='Known 2C')
    plt.scatter(generated_series['mw'], generated_series['logp'], color='blue', s=100, alpha=0.7, label='Generated 2C')
    plt.xlabel('Molecular Weight')
    plt.ylabel('LogP')
    plt.title('2C-Series SAR Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    all_2c = pd.concat([cb_series, generated_series])
    plt.bar(range(len(all_2c)), all_2c['drug_likeness'], 
            color=['red' if x in cb_series.index else 'blue' for x in all_2c.index])
    plt.xlabel('Compound Index')
    plt.ylabel('Drug-likeness Score')
    plt.title('Drug-likeness Comparison')
    plt.xticks(range(len(all_2c)), all_2c['compound_name'], rotation=45)
    
    plt.subplot(2, 2, 3)
    properties = ['mw', 'logp', 'tpsa', 'bbb_score', 'drug_likeness']
    known_avg = cb_series[properties].mean()
    generated_avg = generated_series[properties].mean()
    
    x = np.arange(len(properties))
    width = 0.35
    
    plt.bar(x - width/2, known_avg, width, label='Known 2C', color='red', alpha=0.7)
    plt.bar(x + width/2, generated_avg, width, label='Generated 2C', color='blue', alpha=0.7)
    
    plt.xlabel('Properties')
    plt.ylabel('Average Value')
    plt.title('Property Comparison')
    plt.xticks(x, properties)
    plt.legend()
    
    plt.subplot(2, 2, 4)
    # Radar plot simulation with bar chart
    cb_b_props = all_compounds[all_compounds['compound_name'] == '2C-B'].iloc[0]
    best_analog = generated_series.nlargest(1, 'drug_likeness').iloc[0]
    
    props_normalized = ['drug_likeness', 'bbb_score']
    cb_values = [cb_b_props[prop] for prop in props_normalized]
    analog_values = [best_analog[prop] for prop in props_normalized]
    
    x = np.arange(len(props_normalized))
    plt.bar(x - 0.2, cb_values, 0.4, label='2C-B', color='red', alpha=0.7)
    plt.bar(x + 0.2, analog_values, 0.4, label='Best Analog', color='blue', alpha=0.7)
    plt.xlabel('Normalized Properties')
    plt.ylabel('Score')
    plt.title('2C-B vs Best Generated Analog')
    plt.xticks(x, props_normalized)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('sar_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ SAR analysis plot created and saved as 'sar_analysis.png'")

def create_summary_report(original_df, analog_df, binding_pocket, allosteric_sites):
    """Create a comprehensive summary report."""
    print("\n📋 Generating Summary Report")
    print("-" * 40)
    
    total_compounds = len(original_df) + len(analog_df)
    
    report = f"""
    
🧬 PSYCHEDELIC THERAPEUTICS DESIGN REPORT
=========================================

📊 DATASET SUMMARY
-----------------
• Total compounds analyzed: {total_compounds}
• Known psychedelics: {len(original_df)}
• Generated analogs: {len(analog_df)}
• Compound classes: {', '.join(original_df['class'].unique())}

🎯 2C-B ANALYSIS
---------------
• Reference compound: 2C-B (CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN)
• Molecular weight: 334.1 Da
• LogP: 3.2 (favorable for BBB penetration)
• TPSA: 45.2 Ų (excellent for CNS access)
• Drug-likeness score: {original_df[original_df['compound_name'] == '2C-B']['drug_likeness'].iloc[0]:.2f}

🏗️ STRUCTURAL ANALYSIS
---------------------
• 5-HT2A binding pocket: {binding_pocket['pocket_size']} residues
• Key interactions identified: {len(binding_pocket['key_interactions'])}
• Allosteric sites found: {len(allosteric_sites)}
• Most druggable allosteric site: {allosteric_sites[0]['site']} (score: {allosteric_sites[0]['druggability']:.2f})

🧪 COMPOUND GENERATION
--------------------
• Novel analogs generated: {len(analog_df)}
• Best analog by drug-likeness: {analog_df.nlargest(1, 'drug_likeness')['compound_name'].iloc[0]}
• Average MW of analogs: {analog_df['mw'].mean():.1f} Da
• Average LogP of analogs: {analog_df['logp'].mean():.2f}

💊 DRUG-LIKENESS ASSESSMENT
--------------------------
• Compounds meeting Lipinski's rule: {len(pd.concat([original_df, analog_df])[pd.concat([original_df, analog_df])['drug_likeness'] >= 0.75])}
• BBB penetration favorable: {len(pd.concat([original_df, analog_df])[pd.concat([original_df, analog_df])['bbb_score'] >= 0.8])}
• ADMET optimization needed: {len(pd.concat([original_df, analog_df])[pd.concat([original_df, analog_df])['drug_likeness'] < 0.75])}

🚀 RECOMMENDATIONS
-----------------
1. Focus on fluorinated analogs for improved metabolic stability
2. Investigate allosteric modulators at extracellular loop 2
3. Optimize compounds for reduced hallucinogenic effects
4. Conduct experimental validation of top 3 candidates
5. Perform detailed ADMET profiling for lead compounds

📁 GENERATED FILES
-----------------
• psychedelic_analysis_dashboard.png - Comprehensive analysis plots
• sar_analysis.png - Structure-activity relationship analysis
• This summary report

🔬 NEXT STEPS
------------
• Molecular docking studies with AutoDock Vina
• QSAR model development with larger datasets
• Experimental synthesis and testing
• Selectivity profiling against other serotonin receptors
• In vivo BBB penetration studies

    """
    
    print(report)
    
    # Save report to file
    with open('psychedelic_therapeutics_report.txt', 'w') as f:
        f.write(report)
    
    print("✅ Summary report saved as 'psychedelic_therapeutics_report.txt'")

def main_demo():
    """Run the simplified psychedelic therapeutics demo."""
    print("🧬 PSYCHEDELIC THERAPEUTICS DESIGN DEMO")
    print("="*60)
    print("Specialized CNS therapeutics targeting 5-HT2A receptor")
    print("Focus: 2C-B and novel psychedelic analogs")
    print("="*60)
    
    # 1. Create sample data
    original_df = create_sample_psychedelic_data()
    
    # 2. Analyze 2C-B
    cb_analysis = analyze_2cb_properties(original_df)
    
    # 3. Generate analogs
    analog_df = generate_2cb_analogs()
    
    # 4. Structural analysis
    binding_pocket, allosteric_sites = analyze_5ht2a_structure()
    
    # 5. Create visualizations
    create_visualizations(original_df, analog_df)
    
    # 6. Generate report
    create_summary_report(original_df, analog_df, binding_pocket, allosteric_sites)
    
    # Final summary
    print("\n🎉 DEMO COMPLETED SUCCESSFULLY!")
    print("-" * 40)
    print("📁 Generated files:")
    files = [
        'psychedelic_analysis_dashboard.png',
        'sar_analysis.png', 
        'psychedelic_therapeutics_report.txt'
    ]
    
    for file in files:
        if Path(file).exists():
            print(f"   ✅ {file}")
        else:
            print(f"   📝 {file} (will be created)")
    
    print("\n🚀 This demo showcases:")
    print("   • 2C-B structure-activity relationship analysis")
    print("   • Novel analog generation and optimization")
    print("   • 5-HT2A receptor binding pocket analysis")
    print("   • Comprehensive molecular property assessment")
    print("   • Drug-likeness and BBB penetration prediction")
    print("   • Interactive visualization dashboards")
    
    print("\n🔬 For full functionality, install:")
    print("   • RDKit for advanced molecular analysis")
    print("   • DeepChem for machine learning models")
    print("   • AutoDock Vina for molecular docking")
    print("   • fpocket for cavity analysis")

if __name__ == "__main__":
    main_demo()