from src.cns_optimization import CNSOptimizer
from rdkit import Chem

print("🧠 Testing CNS Optimization...")
optimizer = CNSOptimizer()
mol = Chem.MolFromSmiles("CCc1cc(Br)c(OCc2ccccc2)c(Br)c1CCN")
results = optimizer.calculate_cns_score(mol)

print(f"CNS Score: {results['composite_cns_score']:.3f}")
print(f"Classification: {results['cns_classification']}")
print(f"BBB Score: {results['bbb_score']:.3f}")
print("✅ CNS Optimization Working!")