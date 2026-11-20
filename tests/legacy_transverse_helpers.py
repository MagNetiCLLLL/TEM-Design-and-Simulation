"""Explicit bulk-winding fixtures retained after fine-wire defaults were added.

These fixtures exercise legacy geometry/editing contracts. New installed counts,
complete wire CAD and current-source behavior have independent regression tests.
"""
from copy import deepcopy
from collections.abc import Mapping


def legacy_transverse_document(document):
    def detach(value):
        if isinstance(value,Mapping):return {k:detach(v) for k,v in value.items()}
        if isinstance(value,(tuple,list)):return [detach(v) for v in value]
        return deepcopy(value)
    result=detach(document)
    for part in result['parts']:
        multipole=any(str(field).startswith('multipole_winding_') for field in part)
        prefixes=('transverse_winding_','ideal_transverse_',
                  'multipole_winding_','ideal_multipole_','pole_root_sleeve_')
        removed=set()
        for field in tuple(part):
            if field.startswith(prefixes):
                removed.add(field)
                part.pop(field)
        if multipole and part.get('key') in {'probe_hp1_hexapole','probe_hp2_hexapole'}:
            # Before the multipole-current task these physical N=2 windings
            # had unknown conductor area and deliberately retained bulk CAD.
            field='hexapole_winding_wire_cross_section_mm2'
            part.pop(field,None);removed.add(field)
        metadata=part.get('parameter_metadata',{})
        for field in tuple(metadata):
            if field.startswith(prefixes) or field in removed:
                metadata.pop(field)
        # Keep an explicit empty metadata mapping just as the public winding
        # clear operation does; unrelated metadata is never removed.
    return result


def legacy_transverse_draft(path,*,persist=False):
    from temsim.part_model_document import PartModelDocument
    draft=PartModelDocument(path)
    # Fixture authoring clears the complete optical-unit group atomically in a
    # single validation, rather than repeatedly preparing the whole catalog.
    # Public per-host clear/Undo/save has dedicated input regression coverage.
    draft._commit_component_operation(legacy_transverse_document(draft.document))
    if persist:
        # Only callers' copied temporary fixtures use this branch. Materialize
        # shared definitions into an independent bulk-era test input.
        try:
            draft.save()
        except ValueError as error:
            if 'Save a composed assembly inside its catalog' not in str(error):raise
            from temsim.module_manifest import _atomic_write_text
            _atomic_write_text(draft.path,draft.independent_copy_text())
        draft=PartModelDocument(path)
    return draft
