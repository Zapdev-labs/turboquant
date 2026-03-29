const turboquant = require('./index.js');

console.log('TurboQuant JavaScript API Demo\n');

// Show version
console.log('Version:', turboquant.version);

// Check if available
if (turboquant.isAvailable()) {
  console.log('✓ TurboQuant Python package is installed\n');
  
  // Show help
  console.log('CLI Help:');
  console.log(turboquant.getHelp());
  
  // Example Python code
  console.log('\nExample Python code:');
  console.log(turboquant.getExample());
} else {
  console.log('✗ TurboQuant Python package not found');
  console.log('  Run: pip install turboquant');
}
