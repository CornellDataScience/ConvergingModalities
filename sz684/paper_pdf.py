"""Render the working manuscript with ReportLab (bundled PDF runtime)."""
import html
import re
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,KeepTogether,PageBreak
from reportlab.lib.pagesizes import letter

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'STATE_TRANSFER_PAPER.md';OUTPUT=ROOT/'output/pdf/state_transfer_paper.pdf'


def clean(s):
    for a,b in [('→',' to '),('−','-'),('–','-'),('—','-'),('’',"'"),('“','"'),('”','"'),('\u00a0',' ')]:s=s.replace(a,b)
    # Insert spaces in compact protocol-derived notation without changing numbers.
    return s


def inline(s):
    s=html.escape(clean(s))
    s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s)
    s=re.sub(r'`([^`]+)`',r'<font name="Courier">\1</font>',s)
    return s


def render(source=None, output=None, title='Do State-Change Directions Transfer from Text to Images?'):
    source=SOURCE if source is None else Path(source)
    output=OUTPUT if output is None else Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    styles=getSampleStyleSheet();styles.add(ParagraphStyle(name='BodyPaper',fontName='Times-Roman',fontSize=10.5,leading=14,spaceAfter=8,alignment=TA_LEFT,allowWidows=0,allowOrphans=0))
    styles.add(ParagraphStyle(name='CaptionPaper',fontName='Helvetica',fontSize=8.5,leading=11,spaceAfter=11,textColor=colors.HexColor('#435165')))
    styles['Title'].fontName='Helvetica-Bold';styles['Title'].fontSize=22;styles['Title'].leading=26;styles['Title'].spaceAfter=12
    for k in ['Heading1','Heading2']:
        styles[k].fontName='Helvetica-Bold';styles[k].fontSize=13 if k=='Heading1' else 11;styles[k].leading=17 if k=='Heading1' else 14;styles[k].spaceBefore=12;styles[k].spaceAfter=7;styles[k].keepWithNext=1
    styles.add(ParagraphStyle(name='TablePaper',fontName='Helvetica',fontSize=8,leading=10))
    story=[];lines=source.read_text().splitlines();i=0;width=500
    while i<len(lines):
        line=lines[i].strip()
        if not line:i+=1;continue
        if line.startswith('|'):
            data=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                cells=[c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-:]+',c) for c in cells):data.append([Paragraph(inline(c),styles['TablePaper']) for c in cells])
                i+=1
            count=len(data[0]);widths=([225,70,205] if count==3 else [145]+[(width-145)/(count-1)]*(count-1))
            headers=[cell.getPlainText().lower() for cell in data[0]]
            if count==3 and headers[0] in ('role','stage'):
                widths=[95,180,225]
            elif count==3 and 'first' in headers[1]:
                widths=[260,120,120]
            table=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT',rowSplitRange=(3,-2) if len(data)>=6 else None)
            table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef2f5')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#788697')),('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d8dee5')),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
            story.extend([table,Spacer(1,10)]);continue
        image=re.match(r'!\[(.*?)\]\((.*?)\)',line)
        if image:
            path=image.group(2);w,h=ImageReader(path).getSize();picture=Image(path,width=width,height=width*h/w)
            j=i+1
            while j<len(lines) and not lines[j].strip():j+=1
            if j<len(lines) and lines[j].startswith('Figure '):
                story.append(KeepTogether([picture,Spacer(1,5),Paragraph(inline(lines[j]),styles['CaptionPaper'])]));i=j+1
            else:story.append(picture);i+=1
            continue
        if line.startswith('# '):story.append(Paragraph(inline(line[2:]),styles['Title']))
        elif line.startswith('## '):
            if line=='## References':story.append(PageBreak())
            story.append(Paragraph(inline(line[3:]),styles['Heading1']))
        elif line.startswith('### '):story.append(Paragraph(inline(line[4:]),styles['Heading2']))
        else:
            parts=[line];i+=1
            while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','![')):
                parts.append(lines[i].strip());i+=1
            style=styles['CaptionPaper'] if line.startswith(('Figure ','Working research')) else styles['BodyPaper']
            paragraph=Paragraph(inline(' '.join(parts)),style)
            if line.endswith(':'):paragraph.keepWithNext=1
            story.append(paragraph);continue
        i+=1
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#536174'));canvas.drawString(56,29,'Object-state relation transfer | Working manuscript');canvas.drawRightString(556,29,str(doc.page))
    doc=SimpleDocTemplate(str(output),pagesize=letter,leftMargin=56,rightMargin=56,topMargin=48,bottomMargin=48,title=title,author='Research working draft')
    doc.build(story,onFirstPage=footer,onLaterPages=footer);print(output)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--title',default='Do State-Change Directions Transfer from Text to Images?')
    args=parser.parse_args()
    render(args.source,args.output,args.title)
