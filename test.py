import os

from logzero import logger
from sqlalchemy import create_engine
from sqlalchemy.sql import text
from omegaconf import OmegaConf

from radie.src.extractor import Extractor


def main():

    # load config
    radie_dir = os.environ.get("RADIEPARH")
    config = OmegaConf.load(os.path.join(radie_dir, 'config.yaml'))
    db_config = config.db

    # db conncet
    engine = create_engine(
        f'postgresql://{db_config.user}:{db_config.password}@{db_config.hostname}:{db_config.port}/{db_config.dbname}'
    )

    reports = engine.execute("SELECT * FROM s_t06v_image_inspection_report_v2")

    # instantiate extractor
    extractor = Extractor(do_preprocessing=True,
                          do_split_sentence=True,
                          do_tokenize=True)

    sql_place_holer = ':kanjyaid, :orderno, :groupno, :reportedition, :kensajissidate, :kensajissitime, :objectsequence, :objectname, :objectcertainty'

    for report in reports:
        shoken_outputs = extractor(report['reportshoken'])
        for i, output in enumerate(shoken_outputs):
            output.chunking()
            engine.execute(
                text(
                    f"INSERT INTO structured_radiology_report_table values ({sql_place_holer})"),
                {
                    'kanjyaid': report['kanjyaid'],
                    'orderno': report['orderno'],
                    'groupno': report['groupno'],
                    'reportedition': report['reportedition'],
                    'kensajissidate': report['kensajissidate'],
                    'kensajissitime': report['kensajissitime'],
                    'objectsequence': i,
                    'objectname': output.clinical_object.entity.tokens,
                    'objectcertainty': output.clinical_object.certainty_scale
                })
        logger.info(f'structured model : {shoken_outputs}')


if __name__ == "__main__":
    main()
