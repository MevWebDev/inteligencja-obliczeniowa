import numpy as np
import pandas as pd
from apyori import apriori


data = pd.read_csv('titanic.csv')

items = []
for i in range(data.shape[0]):
    items.append([str(data.values[i,j]) for j in range(1, data.shape[1])])

final_rule = apriori(items, min_support=0.005, min_confidence=0.7)
final_results = list(final_rule)

# Function to extract only survival-related rules
def extract_survival_rules(results):
    survival_rules = []
    
    for result in results:
        support = result.support
        
        for rule in result.ordered_statistics:
            antecedent = list(rule.items_base)
            consequent = list(rule.items_add)
            confidence = rule.confidence
            lift = rule.lift
            
            # Only keep rules where consequent is about survival (0 = died, 1 = survived)
            if 'No' in consequent or 'Yes' in consequent:
                survival_rules.append((antecedent, consequent, support, confidence, lift))
    
    return survival_rules

# Get only survival rules
survival_rules = extract_survival_rules(final_results)

def interpret_survival_rules(rules):
    print("TITANIC SURVIVAL PREDICTION RULES:")
    print("=" * 100)
    
    for i, (antecedent, consequent, support, confidence, lift) in enumerate(rules, 1):
        # Convert antecedent to readable format
        conditions = []
        for item in antecedent:
            if item == '1st':
                conditions.append("1st class")
            elif item == '2nd':
                conditions.append("2nd class") 
            elif item == '3rd':
                conditions.append("3rd class")
            elif item == 'Male':
                conditions.append("male")
            elif item == 'Female':
                conditions.append("female")
            elif item == 'Child':
                conditions.append("child")
            elif item == 'Adult':
                conditions.append("adult")
            else:
                conditions.append(item)
        
        # Convert consequent to survival outcome
        outcome = "SURVIVED" if 'Yes' in consequent else "DIED"
        outcome_color = "✓ SURVIVED" if 'Yes' in consequent else "✗ DIED"
        
        conditions_str = " AND ".join(conditions)
        
        print(f"\n🔍 Rule {i}:")
        print(f"   IF passenger is: {conditions_str}")
        print(f"   THEN they: {outcome_color}")
        print(f"   📊 Accuracy: {confidence:.1%} ({confidence*100:.1f}% of the time)")
        print(f"   📈 Coverage: {support:.1%} ({support*100:.1f}% of all passengers)")
        print(f"   🎯 Strength: {lift:.2f} ({'Very Strong' if lift > 2 else 'Strong' if lift > 1.5 else 'Moderate' if lift > 1.2 else 'Weak'})")

# Display survival rules
print(f"Found {len(survival_rules)} survival-related rules:")
interpret_survival_rules(survival_rules)

# Create summary of key patterns
# Create summary of key patterns
print("\n" + "="*60)
print("📋 SURVIVAL PATTERNS SUMMARY:")
print("="*60)

# Fix: Look for 'No' and 'Yes' instead of '0' and '1'
died_rules = [r for r in survival_rules if 'No' in r[1]]
survived_rules = [r for r in survival_rules if 'Yes' in r[1]]

print(f"\n💀 DEATH PATTERNS ({len(died_rules)} rules):")
for i, (ant, con, sup, conf, lift) in enumerate(died_rules, 1):
    # Clean up the conditions formatting
    conditions = []
    for item in ant:
        if item == '1st':
            conditions.append("1st class")
        elif item == '2nd':
            conditions.append("2nd class") 
        elif item == '3rd':
            conditions.append("3rd class")
        elif item == 'Male':
            conditions.append("Male")
        elif item == 'Female':
            conditions.append("Female")
        elif item == 'Child':
            conditions.append("Child")
        elif item == 'Adult':
            conditions.append("Adult")
        else:
            conditions.append(item)
    
    conditions_str = " + ".join(conditions)
    print(f"   {i}. {conditions_str} → Death (Accuracy: {conf:.1%})")

print(f"\n🌟 SURVIVAL PATTERNS ({len(survived_rules)} rules):")
for i, (ant, con, sup, conf, lift) in enumerate(survived_rules, 1):
    # Clean up the conditions formatting
    conditions = []
    for item in ant:
        if item == '1st':
            conditions.append("1st class")
        elif item == '2nd':
            conditions.append("2nd class") 
        elif item == '3rd':
            conditions.append("3rd class")
        elif item == 'Male':
            conditions.append("Male")
        elif item == 'Female':
            conditions.append("Female")
        elif item == 'Child':
            conditions.append("Child")
        elif item == 'Adult':
            conditions.append("Adult")
        else:
            conditions.append(item)
    
    conditions_str = " + ".join(conditions)
    print(f"   {i}. {conditions_str} → Survival (Accuracy: {conf:.1%})")

# Add some additional analysis
print(f"\n📊 ANALYSIS:")
print(f"Total survival-related rules: {len(survival_rules)}")
print(f"Rules predicting death: {len(died_rules)}")
print(f"Rules predicting survival: {len(survived_rules)}")

if died_rules:
    avg_death_confidence = sum(r[3] for r in died_rules) / len(died_rules)
    print(f"Average confidence for death predictions: {avg_death_confidence:.1%}")

if survived_rules:
    avg_survival_confidence = sum(r[3] for r in survived_rules) / len(survived_rules)
    print(f"Average confidence for survival predictions: {avg_survival_confidence:.1%}")

import matplotlib.pyplot as plt
import numpy as np

# Create visualizations for the rules
def plot_survival_analysis(died_rules, survived_rules):
    
    # Plot 1: Death Rules Analysis
    if died_rules:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Death rules confidence plot
        death_conditions = []
        death_confidences = []
        
        for ant, con, sup, conf, lift in died_rules:
            conditions = []
            for item in ant:
                if item == '1st':
                    conditions.append("1st class")
                elif item == '2nd':
                    conditions.append("2nd class") 
                elif item == '3rd':
                    conditions.append("3rd class")
                elif item == 'Male':
                    conditions.append("Male")
                elif item == 'Female':
                    conditions.append("Female")
                elif item == 'Child':
                    conditions.append("Child")
                elif item == 'Adult':
                    conditions.append("Adult")
                else:
                    conditions.append(item)
            
            death_conditions.append(" + ".join(conditions))
            death_confidences.append(conf * 100)  # Convert to percentage
        
        # Create horizontal bar chart for death rules
        y_pos = np.arange(len(death_conditions))
        bars1 = ax1.barh(y_pos, death_confidences, color='red', alpha=0.7)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(death_conditions, fontsize=8)
        ax1.set_xlabel('Confidence (%)')
        ax1.set_title('💀 Death Prediction Rules', fontsize=14, fontweight='bold')
        ax1.set_xlim(0, 100)
        
        # Add confidence values on bars
        for i, bar in enumerate(bars1):
            width = bar.get_width()
            ax1.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center', fontsize=8)
        
        ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Survival Rules Analysis
    if survived_rules:
        if not died_rules:
            fig, ax2 = plt.subplots(1, 1, figsize=(10, 6))
        
        # Survival rules confidence plot
        survival_conditions = []
        survival_confidences = []
        
        for ant, con, sup, conf, lift in survived_rules:
            conditions = []
            for item in ant:
                if item == '1st':
                    conditions.append("1st class")
                elif item == '2nd':
                    conditions.append("2nd class") 
                elif item == '3rd':
                    conditions.append("3rd class")
                elif item == 'Male':
                    conditions.append("Male")
                elif item == 'Female':
                    conditions.append("Female")
                elif item == 'Child':
                    conditions.append("Child")
                elif item == 'Adult':
                    conditions.append("Adult")
                else:
                    conditions.append(item)
            
            survival_conditions.append(" + ".join(conditions))
            survival_confidences.append(conf * 100)  # Convert to percentage
        
        # Create horizontal bar chart for survival rules
        y_pos2 = np.arange(len(survival_conditions))
        bars2 = ax2.barh(y_pos2, survival_confidences, color='green', alpha=0.7)
        ax2.set_yticks(y_pos2)
        ax2.set_yticklabels(survival_conditions, fontsize=8)
        ax2.set_xlabel('Confidence (%)')
        ax2.set_title('🌟 Survival Prediction Rules', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        
        # Add confidence values on bars
        for i, bar in enumerate(bars2):
            width = bar.get_width()
            ax2.text(width + 1, bar.get_y() + bar.get_height()/2, 
                    f'{width:.1f}%', ha='left', va='center', fontsize=8)
        
        ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('titanic_survival_rules.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot 3: Summary comparison
def plot_summary_comparison(died_rules, survived_rules):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Count rules by passenger characteristics
    death_factors = {'Male': 0, 'Female': 0, '1st class': 0, '2nd class': 0, '3rd class': 0, 'Child': 0, 'Adult': 0}
    survival_factors = {'Male': 0, 'Female': 0, '1st class': 0, '2nd class': 0, '3rd class': 0, 'Child': 0, 'Adult': 0}
    
    # Count death rule factors
    for ant, con, sup, conf, lift in died_rules:
        for item in ant:
            if item == 'Male':
                death_factors['Male'] += 1
            elif item == 'Female':
                death_factors['Female'] += 1
            elif item == '1st':
                death_factors['1st class'] += 1
            elif item == '2nd':
                death_factors['2nd class'] += 1
            elif item == '3rd':
                death_factors['3rd class'] += 1
            elif item == 'Child':
                death_factors['Child'] += 1
            elif item == 'Adult':
                death_factors['Adult'] += 1
    
    # Count survival rule factors
    for ant, con, sup, conf, lift in survived_rules:
        for item in ant:
            if item == 'Male':
                survival_factors['Male'] += 1
            elif item == 'Female':
                survival_factors['Female'] += 1
            elif item == '1st':
                survival_factors['1st class'] += 1
            elif item == '2nd':
                survival_factors['2nd class'] += 1
            elif item == '3rd':
                survival_factors['3rd class'] += 1
            elif item == 'Child':
                survival_factors['Child'] += 1
            elif item == 'Adult':
                survival_factors['Adult'] += 1
    
    # Plot death factors
    factors = list(death_factors.keys())
    death_counts = list(death_factors.values())
    ax1.bar(factors, death_counts, color='red', alpha=0.7)
    ax1.set_title('💀 Death Rule Factors', fontweight='bold')
    ax1.set_ylabel('Number of Rules')
    ax1.tick_params(axis='x', rotation=45)
    
    # Plot survival factors
    survival_counts = list(survival_factors.values())
    ax2.bar(factors, survival_counts, color='green', alpha=0.7)
    ax2.set_title('🌟 Survival Rule Factors', fontweight='bold')
    ax2.set_ylabel('Number of Rules')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('titanic_factors_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Generate the plots
plot_survival_analysis(died_rules, survived_rules)
plot_summary_comparison(died_rules, survived_rules)

print("📊 Plots saved as:")
print("   - titanic_survival_rules.png (individual rule confidences)")
print("   - titanic_factors_comparison.png (factor frequency comparison)")