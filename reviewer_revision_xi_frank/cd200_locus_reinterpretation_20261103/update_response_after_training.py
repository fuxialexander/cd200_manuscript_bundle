#!/usr/bin/env python3
"""Publish local author-review response draft only after negative prediction succeeds."""
from pathlib import Path
import json,hashlib,importlib.util,struct
from zipfile import ZipFile,ZIP_DEFLATED
from xml.etree import ElementTree as ET

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent/'response_package_20261102'

def main():
    assert (HERE/'negative_contrast/metadata.json').exists(), 'Negative inference incomplete'
    assert (HERE/'figures/regenerated_s5b_contrast.pdf').exists(), 'Contrast plot incomplete'
    status=json.loads((HERE/'chrombpnet_negative/job_status.json').read_text())
    assert status['state']=='complete', 'Negative model/evaluation pipeline incomplete'
    old=(PACKAGE/'QA_RESPONSES_HUMAN_ATAC_UPDATED.txt').read_text()
    start=old.index('A (Xi/Frank):',old.index('Q5.'))
    end=old.index('Q7.',start)
    replacement='''A (Xi/Frank):
We have added chromosome-held-out benchmarks in GM12878, K562 and mouse cSCC, with count/profile metrics and paired genomic-block confidence intervals (Fig. R1; response to Comment 1). We have also regenerated the three Cd200 profile comparisons using the original MoDL checkpoint and the newly trained mouse ChromBPNet checkpoint on common genomic positions (Fig. R3). ChromBPNet predictions were averaged over 20 bp to match the original MoDL profile target, and ten bases were excluded at each output edge. Error bars in this selected-locus panel are paired spatial-block bootstrap intervals. Only one completed native mouse fold was recovered, so these intervals describe within-locus resampling rather than variation across model folds; no fold-based hypothesis test is claimed.

We recovered the original source table for the apparently missing Peak 1 bar. Fig. S5A contains the ChromBPNet profile-correlation bar (r = 0.84); the very small bar is in Fig. S5B, where the recorded accessibility contrast is 0.0004167. We have trained a new CD200-negative ChromBPNet model using the same chromosome partition as the positive-cell model and regenerated the contrast using common 2048-bp intervals and consistent state normalization (Fig. R4). Both models recover positive contrasts at these loci while underestimating their magnitude on their native scales. The revised plot labels every numerical value explicitly. Original MoDL differential predictions are shown in their native scale; the empirical multiplier used in the earlier display is documented separately in the source table.

For interpretation, we regenerated count and profile contribution scores from the bias-free component of the new positive-cell ChromBPNet model using its official DeepSHAP workflow and 100 dinucleotide-shuffled references per sequence (Fig. R5). We also tested standalone DeepLIFT Rescale with the same sequences and references. The independent implementations agree numerically for matched scalar targets; DeepSHAP itself uses DeepLIFT-family rules, so this agreement is an implementation check rather than independent biological evidence. Both ChromBPNet's official contributions and MoDL's original profile-deletion analysis place their largest positive Peak 1 score in the TEAD motif-containing region. Checkpoint hashes, commands, reference sequences, target definitions, numerical checks and figure source data are supplied with the analysis.

We will correct the original panel references and remove the unsupported statement that ChromBPNet importance profiles were too fragmented to identify transcription factor motifs. The revised interpretation comparison concerns the three specified loci and shows native signed contribution tracks directly. TF-MoDISco identifies recurring sequence patterns across high-contribution segments; that aggregation step is outside this locus-level comparison. MoDL deletion effects and ChromBPNet contribution scores retain their distinct target definitions and units, which are stated in the legend rather than compared as interchangeable effect sizes.

'''
    text=old[:start]+replacement+old[end:]
    captions=[('R3','Regenerated local profile comparison',HERE/'figures/profile_caption.txt',HERE/'figures/regenerated_s5a_profiles.png'),
              ('R4','Regenerated positive–negative accessibility contrast',HERE/'figures/contrast_caption.txt',HERE/'figures/regenerated_s5b_contrast.png'),
              ('R5','Locus-level interpretation comparison',HERE/'figures/interpretation_caption.txt',HERE/'figures/locus_interpretation_comparison.png')]
    for label,title,caption,png in captions:
        text+='\n\nFig. '+label+'. '+title+'. '+caption.read_text().strip()+'\n'
    txt=PACKAGE/'QA_RESPONSES_INTERPRETATION_UPDATED.txt'
    doc=PACKAGE/'QA_RESPONSES_INTERPRETATION_UPDATED.docx'
    assert not txt.exists() and not doc.exists(), 'Preserve previous rendered drafts'
    txt.write_text(text)
    spec=importlib.util.spec_from_file_location('render_responses',PACKAGE/'make_response_docx.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.render(txt,doc)
    with ZipFile(doc) as z:parts={n:z.read(n) for n in z.namelist()}
    w='http://schemas.openxmlformats.org/wordprocessingml/2006/main'
    relns='http://schemas.openxmlformats.org/package/2006/relationships'
    root=ET.fromstring(parts['word/document.xml']);body=root.find('{'+w+'}body')
    rels=ET.fromstring(parts['word/_rels/document.xml.rels'])
    images={'R2':PACKAGE/'human_atac_tumor_normal_1kb/fig4d_cohorts_mean_signal.png'}
    images.update({label:png for label,title,caption,png in captions})
    inserted=[]
    for idx,p in enumerate(list(body)):
        paragraph=''.join(e.text or '' for e in p.iter('{'+w+'}t'))
        for label,png in images.items():
            if not paragraph.startswith('Fig. '+label+'.'):continue
            raw=png.read_bytes();width,height=struct.unpack('>II',raw[16:24])
            cx=6217920;cy=round(cx*height/width);rid='rId'+label
            drawing=f'''<w:p xmlns:w="{w}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture"><w:r><w:drawing><wp:inline><wp:extent cx="{cx}" cy="{cy}"/><wp:docPr id="{len(inserted)+1}" name="Figure {label}"/><a:graphic><a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture"><pic:pic><pic:nvPicPr><pic:cNvPr id="0" name="{label}.png"/><pic:cNvPicPr/></pic:nvPicPr><pic:blipFill><a:blip r:embed="{rid}"/><a:stretch><a:fillRect/></a:stretch></pic:blipFill><pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr></pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'''
            body.insert(list(body).index(p)+1,ET.fromstring(drawing))
            ET.SubElement(rels,'{'+relns+'}Relationship',{'Id':rid,'Type':'http://schemas.openxmlformats.org/officeDocument/2006/relationships/image','Target':'media/'+label+'.png'})
            parts['word/media/'+label+'.png']=raw
            inserted.append(label)
    assert sorted(inserted)==['R2','R3','R4','R5']
    parts['word/document.xml']=ET.tostring(root,encoding='utf-8',xml_declaration=True)
    parts['word/_rels/document.xml.rels']=ET.tostring(rels,encoding='utf-8',xml_declaration=True)
    ct=ET.fromstring(parts['[Content_Types].xml'])
    ET.SubElement(ct,'{http://schemas.openxmlformats.org/package/2006/content-types}Default',{'Extension':'png','ContentType':'image/png'})
    parts['[Content_Types].xml']=ET.tostring(ct,encoding='utf-8',xml_declaration=True)
    with ZipFile(doc,'w',ZIP_DEFLATED) as z:
        for n,b in parts.items():z.writestr(n,b)
    with ZipFile(doc) as z:
        assert z.testzip() is None
        for n in z.namelist():
            if n.endswith('.xml') or n.endswith('.rels'):ET.fromstring(z.read(n))
        recovered=''.join(e.text or '' for e in ET.fromstring(z.read('word/document.xml')).iter('{'+w+'}t'))
        for line in text.splitlines():
            if line and ' | ' not in line:assert line in recovered
    (HERE/'R1_5_RESPONSE.txt').write_text('Reviewer 1, Comment 5\n\n'+replacement)
    readme=PACKAGE/'README.md'
    content=readme.read_text().replace('QA_RESPONSES_HUMAN_ATAC_UPDATED.docx','QA_RESPONSES_INTERPRETATION_UPDATED.docx',1).replace('QA_RESPONSES_HUMAN_ATAC_UPDATED.txt','QA_RESPONSES_INTERPRETATION_UPDATED.txt',1)
    content+='\n## Cd200 interpretation update — 3 November 2026\n\nThe current Q:A resolves the original Peak1 bar/provenance questions using the archived source table and regenerated official contribution scores. Figures R3–R5 are embedded in Word and supplied as editable PDFs under ../cd200_locus_reinterpretation_20261103/figures/. New negative-state ChromBPNet training and the common-window contrast are complete. One independent native mouse fold was available per state; within-locus spatial-bootstrap intervals are explicitly distinguished from fold variability. No TF-MoDISco was run. Original manuscript assembly and final figure numbering remain pending.\n'
    readme.write_text(content)
    manifest_path=PACKAGE/'package_manifest.json';manifest=json.loads(manifest_path.read_text())
    manifest['cd200_interpretation_update']={'files':{str(p.relative_to(PACKAGE)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [txt,doc]},'validation':'DOCX ZIP/XML and text round-trip; four figures embedded; Word visual rendering unavailable'}
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print('Updated response Word/text with R2–R5 images; original drafts preserved.')

if __name__=='__main__':main()
