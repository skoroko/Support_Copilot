"""Dev-only raw-source literal checks for source-bound reference quotes, offline."""
import argparse, hashlib, json, sys, tempfile
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from app.eval.corpus_suite import digest, file_sha, read_json, write_new

def check(candidate, source_root, catalog_path, output, visual_transcriptions=None):
    import app.services
    app.services.__path__.insert(0,str(source_root / "app/services"))
    from app.services.corpus_html import decode_html
    from app.services.chm_reader import CHMReader
    from bs4 import BeautifulSoup
    from pypdf import PdfReader
    from pylibmspack import ChmArchive
    catalog=read_json(catalog_path)
    contract=read_json(candidate / "dev_reference_contract_v2.json")
    if file_sha(catalog_path)!=contract["catalog_sha256"]:
        raise ValueError("Catalog freeze mismatch")
    locations={l["location_id"]:l for l in catalog["locations"]}
    blocks={b["block_id"]:b for b in catalog["blocks"]}
    reviewed=read_json(visual_transcriptions) if visual_transcriptions else []
    cache, readers, archives={}, {}, {}
    normal=lambda s: "".join(s.split())
    def dom(raw):
        soup=BeautifulSoup(decode_html(raw),"html.parser")
        for hidden in soup(["script","style","noscript"]):hidden.decompose()
        return soup.get_text(" ",strip=True)
    scratch=output.parent/".source-quote-reader"
    scratch.mkdir(parents=True,exist_ok=True)
    previous=tempfile.tempdir;tempfile.tempdir=str(scratch.resolve())
    rows=[]
    try:
        for row in contract["cases"]:
            checks=[]
            for slot in row["claim_slots"]:
                for alternative in slot["alternatives"]:
                    for evidence in alternative.get("evidence",[alternative]):
                        loc=locations[evidence["location_id"]]
                        block=blocks[evidence["block_id"]]
                        attributes=block.get("attributes",{})
                        if attributes.get("method")=="reviewed_visual_transcription":
                            matches=[r for r in reviewed if r.get("approval_status")=="approved"
                                     and r.get("source_hash")==evidence["source_hash"] and r.get("page")==loc.get("page")
                                     and r.get("bbox")==block.get("bbox") and r.get("image_sha256")==attributes.get("image_sha256")
                                     and r.get("review_id")==attributes.get("review_id") and r.get("reviewer")==attributes.get("reviewer")
                                     and r.get("product")==loc.get("product") and r.get("version")==loc.get("version")
                                     and evidence["quote"] in r.get("text","")]
                            checks.append({"slot_id":slot["slot_id"],"block_id":evidence["block_id"],
                                           "location_id":loc["location_id"],"quote_sha256":digest(evidence["quote"]),
                                           "source_hash":evidence["source_hash"],"page":loc.get("page"),
                                           "review_id":attributes.get("review_id"),
                                           "status":"approved_literal_visual_binding" if len(matches)==1 and attributes.get("verified") is True else "visual_approval_binding_missing"})
                            continue
                        path=source_root/"data"/evidence["relative_path"]
                        key=(str(path),loc.get("page"),loc.get("chm_member"))
                        try:
                            if key not in cache:
                                if file_sha(path)!=evidence["source_hash"]:raise ValueError("Source freeze mismatch")
                                if path.suffix.lower()==".pdf":
                                    if str(path) not in readers:readers[str(path)]=PdfReader(path)
                                    text=readers[str(path)].pages[loc["page"]-1].extract_text() or ""
                                elif path.suffix.lower()==".chm":
                                    if str(path) not in archives:archives[str(path)]=ChmArchive(str(path))
                                    archive=archives[str(path)]
                                    try:raw=archive.read(loc["chm_member"],max_size=16*1024*1024)
                                    except FileNotFoundError:
                                        size=next(f.get("size") for f in archive.files(include_system=False) if str(f["name"])==loc["chm_member"])
                                        raw=CHMReader()._read_exact_member_ascii_output(archive,loc["chm_member"],size)
                                    text=dom(raw)
                                else:text=dom(path.read_bytes())
                                cache[key]=text
                            matched=normal(evidence["quote"]) in normal(cache[key])
                            checks.append({"slot_id":slot["slot_id"],"block_id":evidence["block_id"],
                                "location_id":loc["location_id"],"page":loc.get("page"),"chm_member":loc.get("chm_member"),
                                "source_hash":evidence["source_hash"],"quote_sha256":digest(evidence["quote"]),
                                "status":"literal_match" if matched else "structured_or_layout_difference_requires_review"})
                        except Exception as exc:
                            checks.append({"slot_id":slot["slot_id"],"block_id":evidence["block_id"],"status":"failed", "error_type":type(exc).__name__})
            rows.append({"case_id":row["case_id"],"reference_case_sha256":digest(row),"checks":checks,
                         "all_literal":all(c["status"]=="literal_match" for c in checks) if checks else None,
                         "all_source_bound":all(c["status"] in {"literal_match", "approved_literal_visual_binding"} for c in checks) if checks else None})
    finally:tempfile.tempdir=previous
    summary=dict(Counter(c["status"] for row in rows for c in row["checks"]))
    write_new(output,{"method":"machine_independent_raw_source_literal_check","candidate_contract_sha256":digest(contract),
               "catalog_sha256":file_sha(catalog_path),"implementation_sha256":file_sha(Path(__file__)),
               "visual_transcriptions_sha256":file_sha(visual_transcriptions) if visual_transcriptions else None,
               "assurance":"whitespace-only normalization; sourcehash+PDFphysicalpage or exactCHMmember; does not prove semantic sufficiency",
               "summary":summary,"cases":rows})
    print(summary)

if __name__=="__main__":
    p=argparse.ArgumentParser(__doc__)
    for name in ["candidate","source-root","catalog","output"]:p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--visual-transcriptions",type=Path)
    a=p.parse_args();check(a.candidate,a.source_root,a.catalog,a.output,a.visual_transcriptions)
