#!/usr/bin/env python3
"""
Unified evaluation script for running all metrics on generated scenes.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import evaluation classes
from eval.calculate_oob import OutOfBoundaryEvaluator
from eval.sca import SceneClassificationAccuracyEvaluator
from eval.fid_metric import FIDEvaluator
from eval.kl_metric import KLDivergenceEvaluator


class UnifiedEvaluator:
    """Unified evaluator that runs all evaluation metrics."""
    
    def __init__(self, config):
        """
        Initialize the unified evaluator.
        
        Args:
            config (dict): Configuration dictionary containing paths and parameters
        """
        self.config = config
        self.output_dir = config.get('output_directory', 'evaluation_results')
        self.verbose = config.get('verbose', True)
        
        # Create output directory
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # Initialize evaluators
        self.evaluators = {}
        self._initialize_evaluators()
    
    def _initialize_evaluators(self):
        """Initialize all evaluation classes."""
        # Out-of-Boundary Evaluator
        self.evaluators['oob'] = OutOfBoundaryEvaluator()
        
        # Scene Classification Accuracy Evaluator
        sca_config = self.config.get('sca', {})
        self.evaluators['sca'] = SceneClassificationAccuracyEvaluator(
            batch_size=sca_config.get('batch_size', 256),
            num_workers=sca_config.get('num_workers', 0),
            epochs=sca_config.get('epochs', 10),
            device=sca_config.get('device', None)
        )
        
        # FID Evaluator
        fid_config = self.config.get('fid', {})
        self.evaluators['fid'] = FIDEvaluator(
            device=fid_config.get('device', None),
            num_iterations=fid_config.get('num_iterations', 10)
        )
        
        # KL Divergence Evaluator
        kl_config = self.config.get('kl', {})
        self.evaluators['kl'] = KLDivergenceEvaluator(
            class_labels=kl_config.get('class_labels', None)
        )
    
    def run_oob_evaluation(self):
        """Run Out-of-Boundary evaluation."""
        if 'oob' not in self.config or not self.config.get('run_oob', True):
            if self.verbose:
                print("Skipping OOB evaluation...")
            return None
        
        if self.verbose:
            print("=" * 60)
            print("Running Out-of-Boundary (OOB) Evaluation")
            print("=" * 60)
        
        try:
            oob_config = self.config['oob']
            input_folder = oob_config['input_folder']
            
            results = self.evaluators['oob'].evaluate(
                input_folder=input_folder,
                output_file=os.path.join(self.output_dir, 'oob_results.txt'),
                verbose=self.verbose
            )
            
            # Save JSON results as well
            json_output = os.path.join(self.output_dir, 'oob_results.json')
            with open(json_output, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results
            
        except Exception as e:
            print(f"Error in OOB evaluation: {e}")
            return None
    
    def run_sca_evaluation(self):
        """Run Scene Classification Accuracy evaluation."""
        if 'sca' not in self.config or not self.config.get('run_sca', True):
            if self.verbose:
                print("Skipping SCA evaluation...")
            return None
        
        if self.verbose:
            print("=" * 60)
            print("Running Scene Classification Accuracy (SCA) Evaluation")
            print("=" * 60)
        
        try:
            sca_config = self.config['sca']
            
            results = self.evaluators['sca'].evaluate(
                path_to_train_renderings=sca_config['path_to_train_renderings'],
                path_to_test_renderings=sca_config['path_to_test_renderings'],
                path_to_synthesized_renderings=sca_config['path_to_synthesized_renderings'],
                output_directory=self.output_dir,
                verbose=self.verbose
            )
            
            return results
            
        except Exception as e:
            print(f"Error in SCA evaluation: {e}")
            return None
    
    def run_fid_evaluation(self):
        """Run FID/KID evaluation."""
        if 'fid' not in self.config or not self.config.get('run_fid', True):
            if self.verbose:
                print("Skipping FID evaluation...")
            return None
        
        if self.verbose:
            print("=" * 60)
            print("Running FID/KID Evaluation")
            print("=" * 60)
        
        try:
            fid_config = self.config['fid']
            
            results = self.evaluators['fid'].evaluate(
                path_to_real_renderings=fid_config['path_to_real_renderings'],
                path_to_synthesized_renderings=fid_config['path_to_synthesized_renderings'],
                output_directory=self.output_dir,
                temp_dir_base=fid_config.get('temp_dir_base', None),
                verbose=self.verbose
            )
            
            return results
            
        except Exception as e:
            print(f"Error in FID evaluation: {e}")
            return None
    
    def run_kl_evaluation(self):
        """Run KL Divergence evaluation."""
        if 'kl' not in self.config or not self.config.get('run_kl', True):
            if self.verbose:
                print("Skipping KL evaluation...")
            return None
        
        if self.verbose:
            print("=" * 60)
            print("Running KL Divergence Evaluation")
            print("=" * 60)
        
        try:
            kl_config = self.config['kl']
            
            results = self.evaluators['kl'].evaluate(
                gt_scenestate_dir=kl_config['gt_scenestate_dir'],
                synthesized_scenestate_dir=kl_config['synthesized_scenestate_dir'],
                output_directory=self.output_dir,
                create_plot=kl_config.get('create_plot', False),
                verbose=self.verbose
            )
            
            return results
            
        except Exception as e:
            print(f"Error in KL evaluation: {e}")
            return None
    
    def run_all_evaluations(self):
        """Run all evaluations and compile results."""
        start_time = time.time()
        
        if self.verbose:
            print("Starting Unified Evaluation Pipeline")
            print(f"Output directory: {self.output_dir}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
        
        # Run all evaluations
        results = {}
        
        # OOB Evaluation
        results['oob'] = self.run_oob_evaluation()
        
        # SCA Evaluation
        results['sca'] = self.run_sca_evaluation()
        
        # FID Evaluation
        results['fid'] = self.run_fid_evaluation()
        
        # KL Evaluation
        results['kl'] = self.run_kl_evaluation()
        
        # Compile summary
        end_time = time.time()
        execution_time = end_time - start_time
        
        summary = self._compile_summary(results, execution_time)
        
        # Save unified results
        unified_results = {
            'timestamp': datetime.now().isoformat(),
            'execution_time_seconds': execution_time,
            'config': self.config,
            'summary': summary,
            'detailed_results': results
        }
        
        unified_output = os.path.join(self.output_dir, 'unified_evaluation_results.json')
        with open(unified_output, 'w') as f:
            json.dump(unified_results, f, indent=2)
        
        # Create summary report
        self._create_summary_report(summary, execution_time)
        
        if self.verbose:
            print("=" * 60)
            print("Evaluation Pipeline Complete!")
            print(f"Total execution time: {execution_time:.2f} seconds")
            print(f"Results saved to: {self.output_dir}")
            print("=" * 60)
        
        return unified_results
    
    def _compile_summary(self, results, execution_time):
        """Compile a summary of all evaluation results."""
        summary = {
            'execution_time_seconds': execution_time,
            'successful_evaluations': [],
            'failed_evaluations': []
        }
        
        # OOB Summary
        if results['oob'] is not None and 'total_scenes' in results['oob']:
            summary['oob'] = {
                'scene_oob_ratio': results['oob']['scene_oob_ratio'],
                'object_oob_ratio': results['oob']['object_oob_ratio'],
                'total_scenes': results['oob']['total_scenes'],
                'total_objects': results['oob']['total_objects']
            }
            summary['successful_evaluations'].append('oob')
        else:
            summary['failed_evaluations'].append('oob')
        
        # SCA Summary
        if results['sca'] is not None:
            summary['sca'] = {
                'mean_accuracy': results['sca']['mean_accuracy'],
                'std_accuracy': results['sca']['std_accuracy'],
                'num_runs': results['sca']['num_runs']
            }
            summary['successful_evaluations'].append('sca')
        else:
            summary['failed_evaluations'].append('sca')
        
        # FID Summary
        if results['fid'] is not None:
            summary['fid'] = {
                'fid_mean': results['fid']['fid_mean'],
                'fid_std': results['fid']['fid_std'],
                'kid_mean': results['fid']['kid_mean'],
                'kid_std': results['fid']['kid_std'],
                'num_iterations': results['fid']['num_iterations']
            }
            summary['successful_evaluations'].append('fid')
        else:
            summary['failed_evaluations'].append('fid')
        
        # KL Summary
        if results['kl'] is not None:
            summary['kl'] = {
                'kl_divergence': results['kl']['kl_divergence'],
                'num_gt_scenes': results['kl']['num_gt_scenes'],
                'num_syn_scenes': results['kl']['num_syn_scenes']
            }
            summary['successful_evaluations'].append('kl')
        else:
            summary['failed_evaluations'].append('kl')
        
        return summary
    
    def _create_summary_report(self, summary, execution_time):
        """Create a human-readable summary report."""
        report_path = os.path.join(self.output_dir, 'evaluation_summary.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Unified Evaluation Results Summary\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Execution Time: {execution_time:.2f} seconds\n")
            f.write(f"Successful Evaluations: {len(summary['successful_evaluations'])}\n")
            f.write(f"Failed Evaluations: {len(summary['failed_evaluations'])}\n\n")
            
            # OOB Results
            if 'oob' in summary:
                f.write("Out-of-Boundary (OOB) Results:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  Scene OOB Ratio: {summary['oob']['scene_oob_ratio']:.2%}\n")
                f.write(f"  Object OOB Ratio: {summary['oob']['object_oob_ratio']:.2%}\n")
                f.write(f"  Total Scenes: {summary['oob']['total_scenes']}\n")
                f.write(f"  Total Objects: {summary['oob']['total_objects']}\n\n")
            
            # SCA Results
            if 'sca' in summary:
                f.write("Scene Classification Accuracy (SCA) Results:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  Mean Accuracy: {summary['sca']['mean_accuracy']:.4f}\n")
                f.write(f"  Std Deviation: {summary['sca']['std_accuracy']:.4f}\n")
                f.write(f"  Number of Runs: {summary['sca']['num_runs']}\n\n")
            
            # FID Results
            if 'fid' in summary:
                f.write("FID/KID Results:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  FID Score: {summary['fid']['fid_mean']:.4f} ± {summary['fid']['fid_std']:.4f}\n")
                f.write(f"  KID Score: {summary['fid']['kid_mean']:.4f} ± {summary['fid']['kid_std']:.4f}\n")
                f.write(f"  Number of Iterations: {summary['fid']['num_iterations']}\n\n")
            
            # KL Results
            if 'kl' in summary:
                f.write("KL Divergence Results:\n")
                f.write("-" * 30 + "\n")
                f.write(f"  KL Divergence: {summary['kl']['kl_divergence']:.6f}\n")
                f.write(f"  GT Scenes: {summary['kl']['num_gt_scenes']}\n")
                f.write(f"  Synthetic Scenes: {summary['kl']['num_syn_scenes']}\n\n")
            
            # Failed evaluations
            if summary['failed_evaluations']:
                f.write("Failed Evaluations:\n")
                f.write("-" * 30 + "\n")
                for failed in summary['failed_evaluations']:
                    f.write(f"  - {failed}\n")


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Run unified evaluation pipeline for all metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file option
    parser.add_argument('--config', type=str, help='Path to YAML configuration file')

    # Individual path options (alternative to config file)
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    parser.add_argument('--oob_input_folder', type=str, help='Input folder for OOB evaluation')
    parser.add_argument('--sca_train_renderings', type=str, help='Path to real training renderings')
    parser.add_argument('--sca_test_renderings', type=str, help='Path to real test renderings')
    parser.add_argument('--sca_syn_renderings', type=str, help='Path to synthesized renderings')
    parser.add_argument('--fid_real_renderings', type=str, help='Path to real renderings for FID')
    parser.add_argument('--fid_syn_renderings', type=str, help='Path to synthesized renderings for FID')
    parser.add_argument('--kl_gt_scenes', type=str, help='Path to ground truth scene states')
    parser.add_argument('--kl_syn_scenes', type=str, help='Path to synthesized scene states')
    
    # Control which evaluations to run
    parser.add_argument('--skip_oob', action='store_true', help='Skip OOB evaluation')
    parser.add_argument('--skip_sca', action='store_true', help='Skip SCA evaluation')
    parser.add_argument('--skip_fid', action='store_true', help='Skip FID evaluation')
    parser.add_argument('--skip_kl', action='store_true', help='Skip KL evaluation')
    
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--quiet', action='store_true', help='Quiet mode (disable verbose)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Build config from command line arguments
        config = {
            'output_directory': args.output_dir,
            'verbose': args.verbose and not args.quiet,
            'run_oob': not args.skip_oob and args.oob_input_folder is not None,
            'run_sca': not args.skip_sca and all([
                args.sca_train_renderings, args.sca_test_renderings, args.sca_syn_renderings
            ]),
            'run_fid': not args.skip_fid and all([
                args.fid_real_renderings, args.fid_syn_renderings
            ]),
            'run_kl': not args.skip_kl and all([
                args.kl_gt_scenes, args.kl_syn_scenes
            ])
        }
        
        if config['run_oob']:
            config['oob'] = {'input_folder': args.oob_input_folder}
        
        if config['run_sca']:
            config['sca'] = {
                'path_to_train_renderings': args.sca_train_renderings,
                'path_to_test_renderings': args.sca_test_renderings,
                'path_to_synthesized_renderings': args.sca_syn_renderings,
                'batch_size': 256,
                'num_workers': 0,
                'epochs': 10
            }
        
        if config['run_fid']:
            config['fid'] = {
                'path_to_real_renderings': args.fid_real_renderings,
                'path_to_synthesized_renderings': args.fid_syn_renderings,
                'num_iterations': 10
            }
        
        if config['run_kl']:
            config['kl'] = {
                'gt_scenestate_dir': args.kl_gt_scenes,
                'synthesized_scenestate_dir': args.kl_syn_scenes,
                'create_plot': True
            }
    
    # Check if any evaluations are enabled
    evaluations_to_run = [config.get(f'run_{eval}', False) for eval in ['oob', 'sca', 'fid', 'kl']]
    if not any(evaluations_to_run):
        print("Error: No evaluations are configured to run!")
        print("Please provide either a config file or the necessary command line arguments.")
        print("Use --help for more information.")
        return
    
    # Run evaluation
    evaluator = UnifiedEvaluator(config)
    results = evaluator.run_all_evaluations()
    
    print("\nEvaluation complete! Check the output directory for detailed results.")


if __name__ == "__main__":
    main()
