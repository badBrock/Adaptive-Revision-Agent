from pathlib import Path
import json
import sys

class VaultConfig:
    def __init__(self):
        self.config_file = Path("vault_config.json")
        
    def setup_vault(self):
        """One-time vault setup"""
        print("🔧 VAULT SETUP - One-time configuration")
        print("=" * 50)
        
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                print(f"📁 Current vault: {config['vault_path']}")
                
                change = input("Change vault path? (y/n): ").strip().lower()
                if not change.startswith('y'):
                    return config['vault_path']
        
        while True:
            vault_path = input("\nEnter your vault path: ").strip()
            
            if not vault_path:
                print("❌ Please provide a path")
                continue
                
            if not Path(vault_path).exists():
                print(f"❌ Path {vault_path} doesn't exist")
                continue
                
            # Test for .md files
            md_files = list(Path(vault_path).rglob("*.md"))
            if not md_files:
                print(f"❌ No .md files found in {vault_path}")
                continue
                
            print(f"✅ Found {len(md_files)} markdown files")
            break
        
        # Save configuration
        config = {"vault_path": vault_path}
        with open(self.config_file, 'w') as f:
            json.dump(config, f)
            
        print(f"✅ Vault configured and saved!")
        return vault_path
    
    def get_vault_path(self):
        """Get configured vault path"""
        if not self.config_file.exists():
            return self.setup_vault()
            
        with open(self.config_file, 'r') as f:
            config = json.load(f)
            return config['vault_path']

if __name__ == "__main__":
    config = VaultConfig()
    vault_path = config.setup_vault()
    print(f"Vault ready at: {vault_path}")