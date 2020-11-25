import copy
from itertools import chain, groupby, product
from more_itertools import chunked
from rc_utils import Triple

entity_rules = {"Characteristics_descriptor": {"post_target": False, "extract_all": False}, 
                "Size_descriptor": {"post_target": False, "extract_all": False}}

def get_anatomical_relation(object_es, attribute_es):
    ae_es = list(filter(lambda e: e.name == 'Anatomical_entity', attribute_es))
    relation_list = []
    if not ae_es:
        return relation_list
    cat_es = object_es + ae_es
    cat_es.sort(key=lambda x:x.start_idx)
    # cat_es.reverse()
    cat_grouped = map(lambda x: list(x[1]), groupby(cat_es, lambda x: x.name == 'Anatomical_entity'))
    for e_group in chunked(cat_grouped, n=2):
        if len(e_group) == 2:
            source_es, target_es = e_group
            # if (len(source_es) == 1) and (len(target_es) == 1):
            #     obj_e = source_es[0] if source_es[0].name != 'Anatomical_entity' else target_es[0]
            #     attr_e = source_es[0] if not source_es[0].name != 'Anatomical_entity' else target_es[0]               
            #     relation_list.append({'obj': {'idx': obj_e.entity_idx, 'chunk': obj_e.chunk}, 'attr': {'idx': attr_e.entity_idx, 'chunk': attr_e.chunk}, 'relation': 'Anatomical_entity'})
            #     continue
            if source_es[0].name != 'Anatomical_entity':
                continue
            else:
                obj_es = source_es if source_es[0].name != 'Anatomical_entity' else target_es
                attr_es = source_es if not source_es[0].name != 'Anatomical_entity' else target_es
                for obj_e in obj_es:
                    for attr_e in attr_es:
                        # relation_list.append({'obj': {'idx': obj_e.entity_idx, 'chunk': obj_e.chunk}, 'attr': {'idx': attr_e.entity_idx, 'chunk': attr_e.chunk}, 'relation': 'Anatomical_entity'})
                        relation_list.append(Triple(obj=obj_e, attr=attr_e, relation='Anatomical_entity'))
    return relation_list

def get_relation(object_es, attribute_es, target_attribute):
    target_attr_es = list(filter(lambda e: e.name == target_attribute, attribute_es))
    post_target = entity_rules[target_attribute]["post_target"]
    extract_all = entity_rules[target_attribute]["extract_all"]
    relation_list = []
    is_break = False
    if not target_attr_es:
        return relation_list
    if (len(object_es) == 1) and (len(target_attr_es) == 1):
        obj_e = object_es[0]
        attr_e = target_attr_es[0]
        # relation_list.append({'obj': {'idx': obj_e.entity_idx, 'chunk': obj_e.chunk}, 'attr': {'idx': attr_e.entity_idx, 'chunk': attr_e.chunk}, 'relation': target_attribute})
        relation_list.append(Triple(obj=obj_e, attr=attr_e, relation=target_attribute))
        return relation_list
    extracted_attr_list = []
    for obj_e in object_es:
        if post_target:
            attr_es = list(filter(lambda e: e.start_idx > obj_e.end_idx, target_attr_es))  # objより後ろの位置のにfilter
        else:
            attr_es = list(filter(lambda e: e.start_idx < obj_e.end_idx, target_attr_es))  # objより前の位置のにfilter
        for attr_e in attr_es:
            if attr_e.entity_idx not in extracted_attr_list:
                # relation_list.append({'obj': {'idx': obj_e.entity_idx, 'chunk': obj_e.chunk}, 'attr': {'idx': attr_e.entity_idx, 'chunk': attr_e.chunk}, 'relation': target_attribute})  
                relation_list.append(Triple(obj=obj_e, attr=attr_e, relation=target_attribute))
            if not extract_all:
                extracted_attr_list.append(attr_e.entity_idx)
    return relation_list

def get_all_relation(object_es, attribute_es, target_attribute):
    target_es = list(filter(lambda e: e.name == target_attribute, attribute_es))
    relation_list = []
    if not target_es:
        return relation_list
    for obj_e, attr_e in list(product(object_es, target_es)):
        # relation_list.append({'obj': {'idx': obj_e.entity_idx, 'chunk': obj_e.chunk}, 'attr': {'idx': attr_e.entity_idx, 'chunk': attr_e.chunk}, 'relation': target_attribute})  
        relation_list.append(Triple(obj=obj_e, attr=attr_e, relation=target_attribute))     
    return relation_list

def get_certainy_relation(tokens, object_es, attribute_es):
    ced_es = list(filter(lambda e: e.name == 'Certainty_descriptor', attribute_es))
    relation_list = []
    if not ced_es:
        return relation_list
    for obj_e in object_es:
        ced_e = get_target_certainty_descriptor(tokens, obj_e, ced_es)
        if ced_e:
            # relation_list.append({'obj': {'idx': obj_e.entity_idx, 'chunk': obj_e.chunk}, 'attr': {'idx': ced_e.entity_idx, 'chunk': ced_e.chunk}, 'relation': 'Certainty_descriptor'})
            relation_list.append(Triple(obj=obj_e, attr=ced_e, relation='Certainty_descriptor'))
    return relation_list
    
def get_target_certainty_descriptor(tokens, object_e, ced_es, boundary_marker='。'):
    """
    指定されたエンティティ（１つ）のcertainty属性を返す
    """
    object_classes = []
    # 最近傍の「。」までを対象範囲とする。
    tokens_indices = [[i, x] for i,x in enumerate(tokens)]
    boundary_indices = list(filter(lambda x: x[1] == boundary_marker, tokens_indices))
    boundary_indices = list(map(lambda x:x[0], boundary_indices))
    boundary_indices = list(filter(lambda x: x > object_e.end_idx, boundary_indices))
    nearest_boundary_index = min(boundary_indices) if boundary_indices else 999
    targert_ced_es = list(filter(lambda x: x.start_idx < nearest_boundary_index, ced_es))
    # 後ろのエンティティのみを対象とする（この時点で複数存在する可能性もある）
    targert_ced_es = list(filter(lambda x: x.start_idx > object_e.end_idx, targert_ced_es))
    if targert_ced_es:
        # 複数存在する場合、文の末尾のエンティティを取得する（二重否定対応） # この場合、「xx1を認め、xx2を疑う」のようなパターンには対応できない
        # return targert_ced_es[-1]
        # 複数存在する場合、最近傍のエンティティを取得する
        return targert_ced_es[0]
    else:  # implicit positive
        return []
