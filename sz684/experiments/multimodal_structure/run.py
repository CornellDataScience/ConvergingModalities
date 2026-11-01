"""Run as: python -m experiments.multimodal_structure.run --help (from sz684)."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare', help='Validate a records CSV')
    prepare.add_argument('csv'); prepare.add_argument('output')
    legacy = commands.add_parser('import-legacy', help='Reuse state-transfer features')
    legacy.add_argument('source'); legacy.add_argument('output')
    av = commands.add_parser('import-av', help='Split local clips into image/audio channels')
    av.add_argument('csv'); av.add_argument('output')
    download = commands.add_parser('download-av', help='Download a small public VGGSounder pilot')
    download.add_argument('output'); download.add_argument('--concepts', type=int, default=24)
    download.add_argument('--examples', type=int, default=3)
    extract = commands.add_parser('extract', help='Extract a pinned unimodal model')
    extract.add_argument('manifest'); extract.add_argument('models'); extract.add_argument('model'); extract.add_argument('output')
    extract.add_argument('--device', default='cpu'); extract.add_argument('--batch-size', type=int, default=8)
    for name in ['evaluate','directions']:
        command = commands.add_parser(name)
        command.add_argument('manifest'); command.add_argument('output')
        command.add_argument('--cache', action='append', required=True, help='NAME=PATH; repeat per model')
        if name == 'evaluate':
            command.add_argument('--split', choices=['train','validation','test','all'], default='test')
            command.add_argument('--k', type=int, default=3)
            command.add_argument('--draws', type=int, default=199)
        else:
            command.add_argument('--dimensions', type=int, default=8)
    report = commands.add_parser('report'); report.add_argument('results'); report.add_argument('output')
    plan = commands.add_parser('study-plan', help='Freeze the larger sampling plan')
    plan.add_argument('annotations'); plan.add_argument('pilot_log'); plan.add_argument('output')
    plan.add_argument('--concepts',type=int,default=80);plan.add_argument('--examples',type=int,default=15)
    acquire = commands.add_parser('study-acquire', help='Download the prespecified larger sample')
    acquire.add_argument('output');acquire.add_argument('--workers',type=int,default=4)
    finalize = commands.add_parser('study-finalize', help='Finalize reviewed family groups before extraction')
    finalize.add_argument('output')
    for name in ['week7-fit','week8-lock','week8-score']:
        stage=commands.add_parser(name)
        stage.add_argument('manifest')
        if name!='week7-fit': stage.add_argument('fits')
        if name=='week8-score': stage.add_argument('lock')
        stage.add_argument('output')
        stage.add_argument('--cache',action='append',required=True,help='NAME=PATH; repeat per model')
        if name=='week7-fit': stage.add_argument('--dimensions',type=int,nargs='+',default=[8,16,32])
        if name=='week8-lock':
            stage.add_argument('--k',type=int,default=3);stage.add_argument('--draws',type=int,default=999)
    setup=commands.add_parser('setup-week8-models',help='Download pinned objective-diverse backbones')
    setup.add_argument('pilot_specs');setup.add_argument('output')
    full_report=commands.add_parser('week8-report',help='Report saved locked study outputs')
    full_report.add_argument('study');full_report.add_argument('output')
    args = parser.parse_args()
    if args.command=='setup-week8-models':
        from .data.setup_week8_models import setup_models
        setup_models(args.pilot_specs,args.output)
    elif args.command=='week8-report':
        from .reporting.week8 import report_study
        report_study(args.study,args.output)
    elif args.command in ['week7-fit','week8-lock','week8-score']:
        caches={}
        for item in args.cache:
            name,path=item.split('=',1)
            if name in caches: parser.error('Cache names must be unique')
            caches[name]=path
        if args.command=='week7-fit':
            from .training.build_fits import build_fits
            build_fits(args.manifest,caches,args.output,args.dimensions)
        elif args.command=='week8-lock':
            from .evaluation.locked import lock_evaluation
            lock_evaluation(args.manifest,caches,args.fits,args.output,args.k,args.draws)
        else:
            from .evaluation.locked import score_locked
            score_locked(args.manifest,caches,args.fits,args.lock,args.output)
    elif args.command=='study-plan':
        from .data.expanded import plan_study
        plan_study(args.annotations,args.pilot_log,args.output,args.concepts,args.examples)
    elif args.command=='study-acquire':
        from .data.expanded import acquire_study
        acquire_study(args.output,args.workers)
    elif args.command=='study-finalize':
        from .data.expanded import finalize_families
        finalize_families(args.output)
    elif args.command == 'prepare':
        from .data.prepare import prepare
        prepare(args.csv, args.output)
    elif args.command == 'import-legacy':
        from .data.legacy import import_legacy
        import_legacy(args.source, args.output)
    elif args.command == 'import-av':
        from .data.av_data import import_clips
        from .data.prepare import prepare
        csv_path = import_clips(args.csv, args.output)
        prepare(csv_path, Path(args.output)/'manifest.json')
    elif args.command == 'download-av':
        from .data.download_vggsounder import download_pilot
        download_pilot(args.output, args.concepts, args.examples)
    elif args.command == 'extract':
        from .extraction.extract import extract
        extract(args.manifest, args.models, args.model, args.output, args.device, args.batch_size)
    elif args.command in ['evaluate','directions']:
        caches = {}
        for item in args.cache:
            name, path = item.split('=', 1)
            if name in caches:
                parser.error('Cache names must be unique')
            caches[name] = path
        if args.command == 'evaluate':
            from .evaluation.evaluate import evaluate
            evaluate(args.manifest, caches, args.output, args.split, args.k, args.draws)
        else:
            from .evaluation.fit_directions import fit_directions
            fit_directions(args.manifest, caches, args.output, args.dimensions)
    elif args.command == 'report':
        from .reporting.report import report
        report(args.results, args.output)


if __name__ == '__main__':
    main()
