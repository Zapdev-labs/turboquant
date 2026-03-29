#!/usr/bin/env node

/**
 * Simple test script for TurboQuant npm package
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🧪 Testing TurboQuant...\n');

// Create a test numpy file
function createTestFile() {
  const testScript = `
import numpy as np
import sys

# Create test data
data = np.random.randn(100, 128).astype(np.float32)
np.save('test_input.npy', data)
print('Created test_input.npy')
`;
  
  fs.writeFileSync('create_test.py', testScript);
  
  try {
    execSync('python3 create_test.py', { stdio: 'inherit' });
    fs.unlinkSync('create_test.py');
    return true;
  } catch (e) {
    console.error('Failed to create test file');
    return false;
  }
}

// Test CLI
function testCLI() {
  console.log('1. Testing CLI help...');
  try {
    execSync('turboquant --help', { stdio: 'inherit' });
    console.log('✓ CLI help works\n');
  } catch (e) {
    console.error('✗ CLI help failed');
    return false;
  }
  
  console.log('2. Testing quick compression...');
  try {
    execSync('turboquant quick test_input.npy test_output.tq.npz', { stdio: 'inherit' });
    console.log('✓ Quick compression works\n');
  } catch (e) {
    console.error('✗ Quick compression failed');
    return false;
  }
  
  console.log('3. Testing info command...');
  try {
    execSync('turboquant info test_output.tq.npz', { stdio: 'inherit' });
    console.log('✓ Info command works\n');
  } catch (e) {
    console.error('✗ Info command failed');
    return false;
  }
  
  console.log('4. Testing KV analyze...');
  try {
    execSync('turboquant kv-analyze --model-size 7b --seq-len 1000', { stdio: 'inherit' });
    console.log('✓ KV analyze works\n');
  } catch (e) {
    console.error('✗ KV analyze failed');
    return false;
  }
  
  return true;
}

// Cleanup
function cleanup() {
  const files = ['test_input.npy', 'test_output.tq.npz'];
  files.forEach(f => {
    try {
      if (fs.existsSync(f)) fs.unlinkSync(f);
    } catch (e) {}
  });
}

// Run tests
async function main() {
  cleanup();
  
  if (!createTestFile()) {
    process.exit(1);
  }
  
  const success = testCLI();
  
  cleanup();
  
  if (success) {
    console.log('🎉 All tests passed!');
    process.exit(0);
  } else {
    console.log('\n❌ Some tests failed');
    process.exit(1);
  }
}

main().catch(e => {
  console.error('Test error:', e);
  cleanup();
  process.exit(1);
});
